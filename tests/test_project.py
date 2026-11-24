"""CPU tests for research invariants; no model downloads or GPU are required."""

import collections
import glob
import json
import os
import platform
import random
import re
import shutil
import sqlite3
import tempfile
import time
import unittest
import zlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import config
import train as workflow
from config import ExperimentContext, RunOptions
import data as features
from data import gretel_rows, sqlite_ok
import evaluate as metrics
from utils import build_prompt, extract_sql
import utils as cache


def cpu_context(output_dir):
    ctx = ExperimentContext(RunOptions(quick=True, output_dir=Path(output_dir)))
    for name, value in dict(
        os=os,
        np=np,
        pd=pd,
        json=json,
        random=random,
        platform=platform,
        re=re,
        time=time,
        sqlite3=sqlite3,
        collections=collections,
        zlib=zlib,
        glob=glob,
        shutil=shutil,
        binomtest=binomtest,
    ).items():
        setattr(ctx, name, value)
    ctx.torch = SimpleNamespace(
        manual_seed=lambda _: None,
        cuda=SimpleNamespace(
            get_device_name=lambda _: "CPU fixture", is_available=lambda: False
        ),
    )
    ctx.is_bfloat16_supported = lambda: False
    config.configure(ctx)
    return ctx


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ctx = cpu_context(self.directory.name)

    def test_configuration_and_saved_manifest(self):
        self.assertEqual(self.ctx.N_TRAIN, 200)
        self.assertEqual(self.ctx.SEEDS, [3407, 42])
        saved = json.loads((Path(self.directory.name) / "config.json").read_text())
        self.assertEqual(saved["BASE_HP"]["lora_r"], 16)

    def test_gretel_filtering_and_field_mapping(self):
        class Rows(list):
            def shuffle(self, seed):
                return self

        rows = Rows(
            [
                dict(
                    sql="SELECT COUNT(*) FROM students;",
                    sql_context="CREATE TABLE students(id INT);",
                    sql_prompt="How many?",
                    sql_complexity="basic",
                    domain="education",
                ),
                dict(
                    sql="DELETE FROM students",
                    sql_context="CREATE TABLE students(id INT);",
                    sql_prompt="Delete all",
                    sql_complexity="basic",
                    domain="education",
                ),
                dict(
                    sql="SELECT missing FROM students",
                    sql_context="CREATE TABLE students(id INT);",
                    sql_prompt="Missing?",
                    sql_complexity="basic",
                    domain="education",
                ),
            ]
        )
        self.ctx.load_dataset = lambda *args, **kwargs: rows
        self.ctx.sqlite_ok = partial(sqlite_ok, self.ctx)
        kept, scanned, reasons = gretel_rows(self.ctx, "train", 10, 3407, 20)
        self.assertEqual(scanned, 3)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["answer"], "SELECT COUNT(*) FROM students")
        self.assertEqual(sum(reasons.values()), 2)

    def test_cache_roundtrip_keeps_prediction_rows(self):
        cache.initialize_cache(self.ctx)
        frame = pd.DataFrame(
            {"question": ["How many?"], "pred": ["SELECT COUNT(*) FROM t"]}
        )
        cache.save_run(
            self.ctx,
            {"key": "fixture", "eval": {"A_test": {"df": frame, "summary": {}}}},
        )
        restored = cache.load_run(self.ctx, "fixture")
        pd.testing.assert_frame_equal(frame, restored["eval"]["A_test"]["df"])
        self.assertIsNone(cache.load_run(self.ctx, "missing"))

    def test_execution_accuracy_checks_results_and_order(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        con.executescript("CREATE TABLE t(n INT); INSERT INTO t VALUES (1),(2);")
        self.ctx._run = partial(metrics._run, self.ctx)
        self.assertTrue(
            metrics.exec_match(self.ctx, "SELECT SUM(n) FROM t", "SELECT 3", [con])
        )
        self.assertFalse(metrics.exec_match(self.ctx, "SELECT 1", "SELECT 3", [con]))
        self.assertFalse(
            metrics.exec_match(
                self.ctx,
                "SELECT n FROM t ORDER BY n DESC",
                "SELECT n FROM t ORDER BY n",
                [con],
            )
        )
        self.assertIsNone(
            metrics.exec_match(self.ctx, "SELECT 1", "SELECT missing FROM t", [con])
        )

    def test_confidence_intervals_and_mcnemar(self):
        mean, low, high = metrics.bootstrap_ci(self.ctx, [True, True, True])
        self.assertEqual((mean, low, high), (100, 100, 100))
        fixed, broken, p = metrics.mcnemar(self.ctx, [False] * 8, [True] * 8)
        self.assertEqual((fixed, broken), (8, 0))
        self.assertLess(p, 0.05)

    def test_prompt_and_extraction_match_training(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages, kwargs

        row = {"context": "CREATE TABLE t(n INT)", "question": "How many?"}
        self.ctx.tok = Tokenizer()
        self.ctx.SYSTEM = __import__("utils", fromlist=["SYSTEM"]).SYSTEM
        self.ctx._user_msg = partial(metrics._user_msg, self.ctx)
        self.assertEqual(
            build_prompt(self.ctx.tok, row), metrics.build_prompt(self.ctx, row)
        )
        self.assertEqual(
            extract_sql("<think>reason</think>```sql\nSELECT 1;\n```"), "SELECT 1"
        )
        self.assertEqual(
            metrics.normalize(self.ctx, "SELECT  COUNT ( n ) ;"), "select count(n)"
        )

    def test_feature_and_metric_pipeline_on_real_sqlite(self):
        row = {
            "question": "How many students?",
            "context": "CREATE TABLE students(id INTEGER);",
            "answer": "SELECT COUNT(*) FROM students",
        }
        ctx = self.ctx
        ctx.display = lambda value: None
        ctx.sqlite_ok = partial(sqlite_ok, ctx)
        features.initialize_features(ctx)
        self.assertEqual(ctx.difficulty(row["answer"]), "Easy")
        self.assertEqual(ctx.difficulty("SELECT * FROM x JOIN y ON x.id=y.id"), "Hard")
        ctx.train_A = ctx.pool_B = ctx.test_A = ctx.test_B = ctx.tune_A = [row]
        ctx.N_TRAIN = ctx.N_VAL = 0
        self.addCleanup(ctx.close)
        metrics.prepare_evaluation(ctx)
        _, databases, _ = ctx.EVAL_SETS["A_test"]
        frame = ctx.score([row["answer"]], [row["answer"]], [row], databases)
        self.assertTrue(frame.iloc[0]["ex"])
        self.assertTrue(frame.iloc[0]["em"])
        self.assertEqual(frame.iloc[0]["token_f1"], 1.0)
        self.assertEqual(ctx.summarize(frame)["ex"][0], 100.0)

    def test_jsonl_inference_loads_one_model_and_writes_predictions(self):
        from infer import main

        source = Path(self.directory.name) / "questions.jsonl"
        destination = Path(self.directory.name) / "predictions.jsonl"
        source.write_text(
            "\n".join(
                json.dumps({"question": q, "context": "CREATE TABLE t(n INT)"})
                for q in ["Count rows?", "Largest n?"]
            )
        )
        with patch("model.Text2SQL") as factory:
            factory.return_value.predict.side_effect = [
                "SELECT COUNT(*) FROM t",
                "SELECT MAX(n) FROM t",
            ]
            main(
                [
                    "--model",
                    "fixture-adapter",
                    "--input-file",
                    str(source),
                    "--output-file",
                    str(destination),
                ]
            )
            factory.assert_called_once_with("fixture-adapter", 1024, None)
        predictions = [
            json.loads(line) for line in destination.read_text().splitlines()
        ]
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[1]["sql"], "SELECT MAX(n) FROM t")

    def test_workflow_closes_connections_on_failure(self):
        connection = sqlite3.connect(":memory:")

        def failing_stage(ctx):
            ctx.EVAL_SETS = {"A_test": ([], [[connection]], [])}
            raise ValueError("fixture error")

        with patch.object(workflow, "STAGES", [("fixture", failing_stage)]):
            with self.assertRaisesRegex(ValueError, "fixture error"):
                workflow.run(RunOptions())
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
