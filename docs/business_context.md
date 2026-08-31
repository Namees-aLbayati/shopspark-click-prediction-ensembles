# ShopSpark — Business Context

## Industry relevance

Online retail in the United States runs on attention. A typical shopper sees
dozens of product slots on a homepage, category page, or cart page in a single
session. Most slots are ignored. The products that are clicked can influence
whether the shopper stays, buys, or leaves.

Click-through rate (CTR) measures the share of displayed products that shoppers
click:

```text
CTR = clicked impressions / total impressions
```

For mid-size US e-commerce companies, even a one-percentage-point improvement
in recommendation-slot CTR can have a meaningful revenue impact. The supplied
case brief cites the following industry context:

- Triple Whale reports a median paid CTR of approximately 1.77% across the US
  e-commerce brands it tracks.
- WebFX reports an average Google Search CTR of approximately 2.69% for
  e-commerce.
- On-site recommendation widgets can reach approximately 5–10% CTR when they
  are targeted well.

These figures describe different traffic and placement types, so they provide
context rather than directly comparable performance targets for ShopSpark.

Companies such as Amazon, Wayfair, Target, Chewy, and Etsy use product-ranking
systems to decide which product to show to each user, in each slot and context.
This case represents a mid-size US e-commerce company at an earlier stage of
that journey.

## Business context

ShopSpark Retail sells products across eight categories through a website and a
mobile app. Mobile accounts for approximately 77% of traffic. The supplied case
brief describes 8,000 active customers and a catalog of 704 products. Customers
range from new sign-ups to loyal repeat buyers and live across major US cities
and a long tail of smaller locations.

ShopSpark currently fills recommendation slots using simple rules, including:

- Best-selling products in a category
- Most-viewed products from the previous week
- Discounted products

These rules do not fully account for differences between users or browsing
contexts. For example, two shoppers with different ages, interests, cities, and
purchase behavior may receive the same homepage recommendations.

## Business problem

ShopSpark wants to estimate the probability that a user will click a product in
a given context. In a later phase, this probability can be used to rank products
within recommendation slots.

## Analytical objective

Build a binary classification model where:

- One row represents one product impression.
- The target is `clicked`: `1` when clicked and `0` otherwise.
- Inputs can describe the user, product, placement, device, and time context.
- The model output is a click probability that can support product ranking.

## Primary success measures

Because clicks may be uncommon, accuracy alone is not sufficient. Candidate
models should be compared using ranking and probability metrics such as ROC-AUC,
PR-AUC, log loss, and calibration. Business evaluation should also consider CTR
lift in top-ranked recommendation slots through a controlled online experiment.

## Data validation note

The case description and the supplied CSV files should be compared during data
understanding. If counts in the brief differ from the observed data, the
observed values should be reported and the difference documented rather than
silently changing either source.

