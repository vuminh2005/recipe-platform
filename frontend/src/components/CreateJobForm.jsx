import { useState } from "react";
import { createJob } from "../api/jobs";
import { buildRecipeJobPayload } from "../utils/buildRecipeJobPayload";
import {
  createConfigurationFromDefaults,
  getDefaultRecipe,
  getEffectiveParametersForDisplay,
} from "../utils/recipeCatalog";
import { validateRecipeForm } from "../utils/recipeFormValidation";
import AutoMLFields from "./AutoMLFields";
import CatsDogsRecipeFields from "./CatsDogsRecipeFields";
import CommonJobFields from "./CommonJobFields";
import RecipeSelector from "./RecipeSelector";
import TabularRandomForestRecipeFields from "./TabularRandomForestRecipeFields";
import ValidationErrors from "./ValidationErrors";

const RECIPE_RENDERERS = {
  "cats-dogs": CatsDogsRecipeFields,
  "tabular-random-forest": TabularRandomForestRecipeFields,
};

function UnavailableForm({ loading, error, issues }) {
  const message = loading
    ? "Loading public recipes..."
    : error || "No supported public recipes are available.";

  return (
    <section className="panel create-job-form" aria-busy={loading}>
      <div className="panel__header">
        <div>
          <p className="eyebrow">New orchestration run</p>
          <h2>Create Recipe Job</h2>
        </div>
      </div>
      <div className={`alert alert--${loading ? "info" : "danger"}`}>
        {message}
      </div>
      <ValidationErrors
        issues={(issues || []).map((item) => ({ path: "catalog", message: item }))}
        tone="warning"
      />
      <button className="button button--primary" type="button" disabled>
        Run Recipe
      </button>
    </section>
  );
}

function ReadyCreateJobForm({ recipes, catalogIssues, onCreated }) {
  const initialRecipe = getDefaultRecipe(recipes);
  const [selectedRecipeId, setSelectedRecipeId] = useState(
    initialRecipe.recipe_id,
  );
  const [name, setName] = useState(`${initialRecipe.recipe_id}-recipe`);
  const [configuration, setConfiguration] = useState(() =>
    createConfigurationFromDefaults(initialRecipe),
  );
  const [submitting, setSubmitting] = useState(false);
  const [validationIssues, setValidationIssues] = useState([]);
  const [requestIssues, setRequestIssues] = useState([]);

  const selectedRecipe =
    recipes.find((recipe) => recipe.recipe_id === selectedRecipeId) ||
    initialRecipe;
  const RecipeFields = RECIPE_RENDERERS[selectedRecipe.recipe_id];

  function selectRecipe(recipeId) {
    const recipe = recipes.find((item) => item.recipe_id === recipeId);
    if (!recipe || !RECIPE_RENDERERS[recipe.recipe_id]) {
      return;
    }

    setSelectedRecipeId(recipe.recipe_id);
    setConfiguration(createConfigurationFromDefaults(recipe));
    setValidationIssues([]);
    setRequestIssues([]);
  }

  function updateTraining(field, value) {
    setConfiguration((current) => ({
      ...current,
      training: {
        ...current.training,
        [field]: value,
      },
    }));
  }

  function updateAutoML(field, value) {
    setConfiguration((current) => ({
      ...current,
      automl: {
        ...current.automl,
        [field]: value,
      },
    }));
  }

  function updateSearchRange(parameter, endpoint, value) {
    setConfiguration((current) => ({
      ...current,
      automl: {
        ...current.automl,
        search_space: {
          ...current.automl.search_space,
          [parameter]: {
            ...current.automl.search_space?.[parameter],
            [endpoint]: value,
          },
        },
      },
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }

    const form = { name, configuration };
    const clientIssues = validateRecipeForm(form, selectedRecipe);
    setValidationIssues(clientIssues);
    setRequestIssues([]);

    if (clientIssues.length > 0) {
      return;
    }

    setSubmitting(true);

    try {
      const payload = buildRecipeJobPayload(form, selectedRecipe);
      const createdJob = await createJob(payload);
      onCreated?.(createdJob);
    } catch (requestError) {
      setRequestIssues(
        requestError.issues?.length
          ? requestError.issues
          : [{ path: "", message: requestError.message }],
      );
    } finally {
      setSubmitting(false);
    }
  }

  const effectiveParameters = getEffectiveParametersForDisplay(
    selectedRecipe,
    configuration,
  );

  return (
    <form className="panel create-job-form" onSubmit={handleSubmit}>
      <div className="panel__header">
        <div>
          <p className="eyebrow">New orchestration run</p>
          <h2>Create Recipe Job</h2>
        </div>
        <span className="model-pill">{selectedRecipe.display_name}</span>
      </div>

      <RecipeSelector
        recipes={recipes}
        selectedRecipe={selectedRecipe}
        onChange={selectRecipe}
      />
      <CommonJobFields name={name} onChange={setName} />
      <RecipeFields
        recipe={selectedRecipe}
        training={configuration.training}
        onChange={updateTraining}
      />
      <AutoMLFields
        recipe={selectedRecipe}
        automl={configuration.automl}
        effectiveParameters={effectiveParameters}
        onFieldChange={updateAutoML}
        onRangeChange={updateSearchRange}
      />

      <ValidationErrors
        issues={(catalogIssues || []).map((item) => ({
          path: "catalog",
          message: item,
        }))}
        tone="warning"
      />
      <ValidationErrors issues={validationIssues} />
      <ValidationErrors issues={requestIssues} />

      <div className="form-actions">
        <button
          className="button button--primary"
          type="submit"
          disabled={submitting}
        >
          {submitting ? "Creating job..." : "Run Recipe"}
        </button>
      </div>
    </form>
  );
}

export default function CreateJobForm({
  recipes,
  catalogLoading,
  catalogError,
  catalogIssues,
  onCreated,
}) {
  if (catalogLoading || catalogError || !recipes?.length) {
    return (
      <UnavailableForm
        loading={catalogLoading}
        error={catalogError}
        issues={catalogIssues}
      />
    );
  }

  return (
    <ReadyCreateJobForm
      key={recipes.map((recipe) => `${recipe.recipe_id}:${recipe.version}`).join("|")}
      recipes={recipes}
      catalogIssues={catalogIssues}
      onCreated={onCreated}
    />
  );
}
