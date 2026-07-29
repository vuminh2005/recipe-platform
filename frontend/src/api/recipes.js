import { apiRequest } from "./client";

export function listRecipes() {
  return apiRequest("/api/recipes");
}

export function getRecipe(recipeId) {
  return apiRequest(`/api/recipes/${encodeURIComponent(recipeId)}`);
}
