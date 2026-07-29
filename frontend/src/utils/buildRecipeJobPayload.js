import { buildCatsDogsJobPayload } from "./buildCatsDogsJobPayload.js";
import { buildTabularRandomForestJobPayload } from "./buildTabularRandomForestJobPayload.js";

const PAYLOAD_BUILDERS = {
  "cats-dogs": buildCatsDogsJobPayload,
  "tabular-random-forest": buildTabularRandomForestJobPayload,
};

export function buildRecipeJobPayload(form, recipe) {
  const builder = PAYLOAD_BUILDERS[recipe?.recipe_id];

  if (!builder) {
    throw new Error(
      `No payload builder is available for ${recipe?.recipe_id || "this recipe"}.`,
    );
  }

  return builder(form, recipe);
}
