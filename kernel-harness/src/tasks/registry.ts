import type { TaskScenario } from "../types.js";
import { airbnbRedExteriorSanFrancisco } from "./scenarios/airbnb.js";
import { craigslistSfRentalsUnder3000 } from "./scenarios/craigslist.js";
import { doordashSearchRestaurant, ubereatsFindMenuItem, ubereatsSearchRestaurant } from "./scenarios/ubereats.js";

const taskRegistry = new Map<string, TaskScenario>([
  [ubereatsSearchRestaurant.id, ubereatsSearchRestaurant],
  [ubereatsFindMenuItem.id, ubereatsFindMenuItem],
  [doordashSearchRestaurant.id, doordashSearchRestaurant],
  [airbnbRedExteriorSanFrancisco.id, airbnbRedExteriorSanFrancisco],
  [craigslistSfRentalsUnder3000.id, craigslistSfRentalsUnder3000]
]);

export function listTasks(): TaskScenario[] {
  return [...taskRegistry.values()];
}

export function getTask(taskId: string): TaskScenario {
  const task = taskRegistry.get(taskId);
  if (!task) {
    throw new Error(`Unknown task: ${taskId}`);
  }
  return task;
}
