import { formatKey } from "../utils/format";

function RecipeSummary({ recipe }) {
  const objective = recipe.objective;

  return (
    <div className="recipe-summary">
      <div>
        <span>Task</span>
        <strong>{formatKey(recipe.task_type)}</strong>
      </div>
      <div>
        <span>Framework</span>
        <strong>{formatKey(recipe.framework || "Not applicable")}</strong>
      </div>
      <div>
        <span>Model</span>
        <strong>{recipe.model || "Not applicable"}</strong>
      </div>
      <div>
        <span>Objective</span>
        <strong>
          {objective
            ? `${objective.name} · ${objective.direction}`
            : "Not applicable"}
        </strong>
      </div>
      <div>
        <span>AutoML</span>
        <strong>{recipe.supports_automl ? "Supported" : "Not supported"}</strong>
      </div>
    </div>
  );
}

export default function RecipeSelector({
  recipes,
  selectedRecipe,
  onChange,
}) {
  return (
    <div className="form-section form-section--first">
      <h3>Recipe Template</h3>

      <label className="field">
        <span>Available public recipe</span>
        <select
          value={selectedRecipe.recipe_id}
          onChange={(event) => onChange(event.target.value)}
        >
          {recipes.map((recipe) => (
            <option key={recipe.recipe_id} value={recipe.recipe_id}>
              {recipe.display_name}
            </option>
          ))}
        </select>
      </label>

      <div className="recipe-description">
        <strong>{selectedRecipe.display_name}</strong>
        <p>{selectedRecipe.description}</p>
        <small>
          Recipe ID: {selectedRecipe.recipe_id} · Version{" "}
          {selectedRecipe.version}
        </small>
      </div>

      <RecipeSummary recipe={selectedRecipe} />
    </div>
  );
}
