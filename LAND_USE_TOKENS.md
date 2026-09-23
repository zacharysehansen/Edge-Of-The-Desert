# Land-Use Tokens — a proposed vocabulary for the table

**For discussion with the installation team. Nothing here is built yet.**
Written 2026-09-22. Companion to [PHASE4.md](PHASE4.md), which is the engineering plan; this
document is the part that needs your agreement before any of it is worth building.

---

## 1. What a token is, in the model

The table reads a physical object placed on a parcel. To the model, that is one sentence:

> **A token changes what a piece of land *is*, and every consequence follows from that.**

Not "urbanization goes up by 3 %." A token says *these acres, here, were desert and are now
houses* — and the model works out what that does to the water underground, the water in the river,
the green on the ground, the fire risk at the edge of town, and the birds.

This matters because of something we measured and then nearly hid. Paving land drops the greenness
of **that land** by about 16 %. Averaged across eight counties, the same act reads as **0.8 points
out of 100** — which on a screen looks like *nothing happened*. The land-use change is real; the
denominator was wrong. Phase 4 fixes the denominator by giving the model a **Tucson-sized local
area** as well as the regional one, so a token placed on the table changes something you can see
and hear.

It also matters because **where** you put it will now mean something. A subdivision inside the
Tucson basin draws on the Tucson aquifer. One outside it does not.

---

## 2. The six proposed tokens

Chosen so that **every token has an opposite**. The table should always pose a trade-off, never a
one-way ratchet.

| Token | What it is | Its opposite |
|---|---|---|
| **Subdivision** | Low-density housing on open land | Dense infill |
| **Dense infill** | The same number of people, far fewer acres | Subdivision |
| **Farm** | New or expanded irrigated cropland | Retire farm |
| **Retire farm** | Cropland taken out of production | Farm |
| **Recharge basin** | Water put back into the ground — CAP and treated effluent | Any pumping token |
| **Riparian restoration** | Bringing a stretch of river back to life | Any pumping token |

### Why a restorative token is not optional

> **Confirmed 2026-09-22, and more strongly than expected.** We measured the Tucson-area monitoring
> wells in our own data: the water table has been **rising about 1.9 feet a year** since 2000,
> because the city recharges Colorado River water and treated effluent into the ground. Recharge is
> not a hopeful gesture in this valley — **it is the thing currently holding the aquifer up.** And
> the state's own population projections put Pima County almost flat to 2060 (+4.6 % on the medium
> series; the low series *declines*). So the century-scale danger here is not a growth boom. It is
> that the river feeding the recharge is projected to shrink. A table without a recharge token
> cannot tell that story at all.


**Recharge basins and river restoration are the only two tokens that make things better.** Without
at least one of them, every session ends the same way — everyone pushes, the aquifer falls, the
piece becomes a doom loop and stops being a land-planning model.

They are also not a courtesy. Tucson genuinely recharges Colorado River water and treated effluent
into the ground, and the effluent-fed Santa Cruz downtown is a real, visible, locally famous
restoration — the *Living River* reports in our repository document exactly this reach. A visitor
from Tucson will recognise both tokens.

---

## 3. What each token does

Five output families. **↑ better / ↓ worse**, from the point of view of the desert.

| Token | Water underground | Water in the river | Green on the ground | Fire | Wildlife |
|---|---|---|---|---|---|
| **Subdivision** | ↓ pumping for new homes | ↑ *more runoff*, ↓ less recharge | ↓ paved | ↓ more ignitions at the town edge | ↓ habitat lost |
| **Dense infill** | ↓ but far less per person | ~ small | ~ small | ~ small | ~ small |
| **Farm** | ↓↓ **the single largest draw** | ↓ | ↑↑ **crops are the greenest thing here** | ? | ↓ |
| **Retire farm** | ↑↑ | ↑ | ↓ | ? | ↑ |
| **Recharge basin** | ↑↑ | ↑ locally | ↑ locally | ~ | ↑ |
| **Riparian restoration** | ~ | ↑ | ↑↑ locally | ~ | ↑↑ |

**The two tensions worth designing around**, because they are where the piece gets interesting:

- **Farming is the greenest and the thirstiest thing on the table.** Add a farm and vegetation
  improves while the aquifer falls. Retire it and the aquifer recovers while the land goes brown.
  There is no setting where both get better. This is measured, not editorialised.
- **Density versus sprawl.** The same population housed two ways: one paves five times the land and
  puts houses against the fuel at the edge of town, the other does not. Same people, different
  desert.

---

## 4. What we actually know, and how well

Our project tags every number by how it was arrived at. Nothing gets stated more confidently than
it deserves, and the installation will show these tags.

| Path | How we know | Confidence |
|---|---|---|
| Paving → greenness | **Measured** in our own satellite data: ~131,000 pixels, each compared against its own past, 2000–04 vs 2019–23 | **strong** |
| *What* gets paved matters enormously | **Measured**: paving bare desert costs almost nothing (−0.003); paving cropland or riverside costs **66× more** (−0.179) | **strong, and it is the story** |
| Farming → greenness | **Measured**, and it matched the textbook value to within 3 % | **strong** |
| Pumping → falling water table | **Measured** on 12-month horizons, but with a real spread | **moderate — shown as a range** |
| People → household water use | **Measured** from our own data (~129 gallons per person per day) | **strong** |
| Colorado River shortage → Arizona cuts | **Transcribed from the federal Drought Contingency Plan** | **strong** |
| Development → river flow | Reasoned from runoff behaviour; **our attempt to measure it returned nothing** | **weak — declared, shown as such** |
| Recharge basins → aquifer | Physically certain in direction; the local number is **not yet sourced** | **to be sourced** |
| River restoration → wildlife | **Not currently in the model at all** — new work | **to be built** |
| Development → fire | **Being measured now**, from 40 years of fire perimeters and 30 years of land cover | **unknown until it is** |

**On that last row, a commitment made in advance:** if the fire measurement comes back with no
real signal, **we will say so and the fire card will respond only to climate.** We will not invent
a number to make a card move. Regardless of the result, the table will carry an honest
**"how much of what you just built sits in the fire-prone edge"** readout, which is pure geometry
and always responds to what you place.

---

## 5. What we need from you

These are your decisions, not ours. Everything downstream recomputes from them, so the model will
carry them as adjustable parameters until you answer.

1. **How many token *types* will exist in hardware?** Six is a proposal. If the fabrication budget
   is three, tell us which three and we will build those — our recommendation for a three-token
   set is **subdivision, farm, recharge basin**, because it preserves both tensions.
2. **How many parcels does the RFID grid resolve?** This sets **how many acres one token is**, and
   therefore how big a single placement feels. We will compute the acres from the Tucson basin's
   real area once we know the grid.
3. **Does a token have an intensity, or is it binary?** One "subdivision" object, or a dial /
   stacking / different sizes? Binary is simpler and probably better for a table people share.
4. **Can tokens be removed mid-session, and should removal undo the consequence?**
   **Our strong recommendation: removal stops the pumping but does not refill the aquifer.** A
   reversible aquifer is a lie, and the irreversibility is what makes several people acting at once
   actually matter.
5. **Does the physical landscape depict a specific real place?** If the model is a particular reach
   of the Santa Cruz, we will match the model's geography to it exactly rather than approximating
   with a watershed boundary.
6. **For sound:** every output will arrive with a **declared full range** and a **provenance flag**
   (measured / modelled / declared / extrapolated), so a mapping can be tuned once rather than
   re-scaling itself every session. Tell us what shape is easiest to consume — continuous stream
   over OSC, or polled.

---

## 6. The story the data tells, which is not the obvious one

Worth knowing before you design the visuals and the sound, because it changes what the piece is
about:

- **Tucson is not projected to boom.** State projections have Pima County at roughly today's
  population in 2060.
- **The aquifer is currently recovering**, at about two feet a year, because of recharge.
- **That recovery depends on the Colorado River**, which is projected to deliver less.

So the failure this piece can honestly show is not a city that outgrows its water. It is **a desert
city kept alive by a river that is shrinking** — and the moment the recharge stops, the recovery
reverses. Arizona law gives that moment a precise name: groundwater must be available for
**100 years** at a depth of no more than **1,000 feet** for a new subdivision to be approved inside
the Tucson management area. Tucson's monitored wells sit around **200 feet** today. The piece can
show the gap between those two numbers opening and closing as people act on the table.

That is also why the time control matters so much. None of this is visible in a month.

## 7. Two things about time

The model is being rebuilt so the table can **run forward** — not three years, but as far as anyone
wants to look.

- **Time runs while people play.** Consequences arrive *late*. Someone adds farms in simulated
  2035 and the water table keeps falling into the 2060s after they have walked away. That delay is
  the truest thing the piece can teach, and it is only possible because time runs.
- **Slow tokens need time to be visible.** A recharge basin does almost nothing in a month and a
  great deal in forty years. The time control is not a feature bolted onto the side — for half the
  vocabulary, **it is what makes the token perceptible at all.**

A note on honesty at long horizons: our best data runs about 25 years for vegetation and 45 for
rivers. The model will keep projecting past that and will **mark the projection as beyond the
evidence** when it does. We would like that visible in the piece rather than buried — a horizon
past which the image or the sound is openly less certain. That is a design opportunity, and it is
yours more than ours.
