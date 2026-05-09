import type { TaskScenario } from "../../types.js";

export const airbnbRedExteriorSanFrancisco: TaskScenario = {
  id: "airbnb_red_exterior_san_francisco",
  title: "Find Airbnb listings with red exterior homes in San Francisco",
  startUrl: "https://www.airbnb.com/",
  goal:
    "Find Airbnb listings in San Francisco where the visible listing photos show homes with red-painted exteriors. Return listing names and URLs when possible.",
  allowedDomains: ["airbnb.com", "www.airbnb.com"],
  successCriteria: [
    "Airbnb San Francisco search results or listing pages are visible.",
    "At least one listing with a visibly red-painted exterior has been identified.",
    "No booking, reserve, payment, login, or message flow has been opened."
  ],
  stopBefore: [
    "reserve",
    "book",
    "checkout",
    "payment",
    "request to book",
    "confirm and pay",
    "log in",
    "sign up",
    "message host"
  ],
  contextForFineTunedModel: [
    "Use Airbnb search for San Francisco stays.",
    "Focus on visible listing card photos or listing photo galleries.",
    "A valid result must visibly show a red-painted exterior, not just red decor or a red icon.",
    "If exact URL is unavailable from the card, open the listing in a non-booking page and copy the URL.",
    "Stop before reserve, checkout, login, messaging, or payment."
  ],
  maxSteps: 45
};
