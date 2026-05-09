import type { TaskScenario } from "../../types.js";

export const ubereatsSearchRestaurant: TaskScenario = {
  id: "ubereats_search_restaurant",
  title: "Search for a restaurant on UberEats",
  startUrl: "https://www.ubereats.com/",
  goal: "Find restaurant options for a requested cuisine and stop after opening a restaurant page.",
  allowedDomains: ["ubereats.com", "www.ubereats.com"],
  successCriteria: [
    "A restaurant detail or menu page is visible.",
    "No checkout, order submission, or payment step has been opened."
  ],
  stopBefore: ["checkout", "place order", "payment", "submit order", "confirm purchase"],
  contextForFineTunedModel: [
    "Prefer navigation paths that use visible search inputs and restaurant result cards.",
    "If location, login, or payment walls appear, stop and report that human setup is required.",
    "Do not click checkout or submit-order controls."
  ],
  maxSteps: 30
};

export const ubereatsFindMenuItem: TaskScenario = {
  id: "ubereats_find_menu_item",
  title: "Find a menu item on UberEats",
  startUrl: "https://www.ubereats.com/",
  goal: "Find a specific food item from a restaurant menu and stop before adding payment or checkout actions.",
  allowedDomains: ["ubereats.com", "www.ubereats.com"],
  successCriteria: [
    "A matching menu item or close substitute is visible.",
    "The agent has not submitted an order."
  ],
  stopBefore: ["checkout", "place order", "payment", "submit order", "confirm purchase"],
  contextForFineTunedModel: [
    "Use menu search or category navigation if available.",
    "Treat checkout, order, and payment buttons as stop points.",
    "Summarize blockers instead of bypassing account or location prompts."
  ],
  maxSteps: 40
};

export const doordashSearchRestaurant: TaskScenario = {
  id: "doordash_search_restaurant",
  title: "Search for a restaurant on DoorDash",
  startUrl: "https://www.doordash.com/",
  goal: "Find restaurant options for a requested cuisine and stop after opening a restaurant or store page.",
  allowedDomains: ["doordash.com", "www.doordash.com"],
  successCriteria: [
    "A restaurant, store, or menu detail page is visible.",
    "No checkout, order submission, or payment step has been opened."
  ],
  stopBefore: ["checkout", "place order", "payment", "submit order", "confirm purchase"],
  contextForFineTunedModel: [
    "If an address prompt appears, use the public landmark Ferry Building, San Francisco.",
    "If autocomplete suggestions appear, select the first suggestion matching Ferry Building.",
    "After cuisine search results load, open the first visible restaurant or store result card.",
    "Do not click checkout, cart, payment, or order submission controls."
  ],
  maxSteps: 35
};
