from __future__ import annotations

import json
import unittest

import httpx
from fastapi import FastAPI

from backend.app.recipe_catalog import router


class RecipeCatalogApiTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = FastAPI()
        cls.app.include_router(router)

    async def get(self, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.get(path)

    async def test_public_list_contains_cats_dogs_but_not_hello(self) -> None:
        response = await self.get("/api/recipes")

        self.assertEqual(response.status_code, 200)
        definitions = response.json()
        self.assertEqual(
            [definition["recipe_id"] for definition in definitions],
            ["cats-dogs"],
        )

    async def test_public_detail_contains_complete_supported_contract(
        self,
    ) -> None:
        response = await self.get("/api/recipes/cats-dogs")

        self.assertEqual(response.status_code, 200)
        definition = response.json()
        self.assertEqual(definition["version"], "1.0")
        self.assertEqual(definition["visibility"], "public")
        self.assertEqual(
            definition["task_type"],
            "binary_image_classification",
        )
        self.assertEqual(definition["framework"], "tensorflow_keras")
        self.assertEqual(definition["model"], "mobilenet_v2")
        self.assertEqual(definition["supported_algorithms"], ["random"])
        self.assertEqual(
            definition["objective"],
            {"name": "val_auc", "direction": "maximize"},
        )
        training_names = {
            field["name"]
            for field in definition["configurable_training_fields"]
        }
        self.assertEqual(
            training_names,
            {
                "image_size",
                "trial_epochs",
                "final_epochs",
                "batch_size",
                "dense_units",
                "trainable_backbone",
            },
        )
        self.assertNotIn("tiny_cnn", json.dumps(definition))
        self.assertNotIn('"epochs"', json.dumps(definition))

    async def test_internal_hello_is_not_a_public_catalog_resource(
        self,
    ) -> None:
        response = await self.get("/api/recipes/hello")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Recipe not found"})

    async def test_unknown_recipe_has_clear_not_found_response(self) -> None:
        response = await self.get("/api/recipes/not-built-in")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Recipe not found"})

    async def test_public_metadata_contains_no_private_execution_fields(
        self,
    ) -> None:
        definition = (await self.get("/api/recipes/cats-dogs")).json()
        serialized = json.dumps(definition).lower()
        for forbidden in (
            "secret",
            "credential",
            "access_key",
            "pipeline_path",
            "trainer_image",
            "container",
            "podspec",
            "command",
            "endpoint_url",
            "service_url",
            "dataset_uri",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    async def test_fields_include_frontend_metadata(self) -> None:
        definition = (await self.get("/api/recipes/cats-dogs")).json()
        fields = (
            definition["configurable_training_fields"]
            + definition["configurable_automl_fields"]
            + definition["configurable_search_space"]
        )
        required_keys = {
            "name",
            "label",
            "description",
            "type",
            "required",
            "default",
            "minimum",
            "maximum",
            "step",
            "exclusive_minimum",
            "exclusive_maximum",
            "options",
        }
        for field in fields:
            with self.subTest(field=field["name"]):
                self.assertEqual(set(field), required_keys)


if __name__ == "__main__":
    unittest.main()
