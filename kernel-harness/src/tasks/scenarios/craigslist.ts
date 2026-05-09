import type { TaskScenario } from "../../types.js";

export const craigslistSfRentalsUnder3000: TaskScenario = {
  id: "craigslist_sf_rentals_under_3000",
  title: "Find San Francisco Craigslist rentals under $3000",
  startUrl: "https://sfbay.craigslist.org/search/sfc/apa?max_price=3000&availabilityMode=0&sale_date=all+dates",
  goal:
    "Find three San Francisco Craigslist housing listings for rent under $3000/month. For each, return title, monthly price, listing URL, public contact method or reply link if visible, and address or general location if visible.",
  allowedDomains: ["craigslist.org", "sfbay.craigslist.org"],
  successCriteria: [
    "Three San Francisco rental listings under $3000/month are identified.",
    "Each listing includes a title, price, URL, contact method or public reply mechanism, and address or general location when visible.",
    "No login, application, message sending, payment, or form submission is performed."
  ],
  stopBefore: [
    "send",
    "submit",
    "apply",
    "payment",
    "pay",
    "sign in",
    "log in",
    "call",
    "text",
    "schedule tour"
  ],
  contextForFineTunedModel: [
    "Use only public Craigslist listing pages and visible listing information.",
    "It is okay to open listing pages and click reply to reveal public Craigslist contact mechanisms, but do not send a message, call, text, log in, or submit any form.",
    "Prefer listings that visibly show San Francisco neighborhood, address, or map/general location.",
    "Return only three listings that are under $3000/month."
  ],
  maxSteps: 60
};
