"""Convert jackdaw raw_state into the view HeuristicBot.act() consumes.

This is the demonstration bridge for behavior-cloning: the heuristic only
*labels* recorded card-level states (never runs inside the trained policy),
so its decisions can supervise the pointer network across the exact
observation distribution training uses.

Only the fields the heuristic reads are translated.  Defensive throughout:
engine objects are accessed by attribute, serialized dicts by key, whichever
appears.
"""

from __future__ import annotations

_RANK_SHORT = {"ACE": "A", "KING": "K", "QUEEN": "Q", "JACK": "J", "TEN": "T",
               "NINE": "9", "EIGHT": "8", "SEVEN": "7", "SIX": "6",
               "FIVE": "5", "FOUR": "4", "THREE": "3", "TWO": "2"}

_PHASE = {
    "blind_select": "BLIND_SELECT",
    "selecting_hand": "SELECTING_HAND",
    "round_eval": "ROUND_EVAL",
    "shop": "SHOP",
    "pack_opening": "SMODS_BOOSTER_OPENED",
}


def _short_rank(name: str) -> str:
    return _RANK_SHORT.get(name, name[:1].upper())


def _attr(obj, name, default=None):
    """Attribute-or-key access, whichever the object supports."""
    if obj is None:
        return default
    v = getattr(obj, name, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(name, default)
    return default if v is None else v


def _cards_of(area):
    """Card list from a CardArea object, dict wrapper, or bare list."""
    if area is None:
        return []
    if isinstance(area, list):
        return area
    cards = getattr(area, "cards", None)
    if cards is None and isinstance(area, dict):
        cards = area.get("cards")
    return cards or []


def _card_value_dict(card) -> dict:
    """{value: {rank, suit}, ...} shape poker.best_play expects."""
    rank = _attr(_attr(card, "base"), "rank") or _attr(card, "rank")
    suit = _attr(_attr(card, "base"), "suit") or _attr(card, "suit")
    rank_s = str(getattr(rank, "name", rank) or "").upper()
    suit_s = str(getattr(suit, "name", suit) or "").upper()
    return {"value": {"rank": _RANK_SHORT.get(rank_s, rank_s[:1]),
                      "suit": suit_s[:1]},
            "_card": card}


def _hands_view(levels) -> dict | None:
    """HandLevels -> {name: {chips, mult, level}} for poker.best_play."""
    if levels is None:
        return None
    out = {}
    hands = getattr(levels, "_hands", None) or {}
    for ht, st in hands.items():
        name = getattr(ht, "value", ht)
        out[str(name)] = {"chips": float(getattr(st, "chips", 0) or 0),
                          "mult": float(getattr(st, "mult", 0) or 0),
                          "level": int(getattr(st, "level", 1) or 1)}
    return out


def blind_view(gs) -> dict | None:
    blind = gs.get("blind")
    if blind is None:
        return None
    name = str(_attr(blind, "name") or "")
    chips = float(_attr(blind, "chips", 0) or 0)
    return {"status": "CURRENT", "name": name, "score": chips}


def build_state(raw: dict) -> dict:
    """jackdaw raw_state -> intent-style state dict for HeuristicBot."""
    phase_raw = str(_attr(raw, "phase") or "")
    phase = _PHASE.get(phase_raw.lower(), phase_raw.upper())

    rr = raw.get("round_resets") or {}
    cur = raw.get("current_round") or {}

    hand_area = raw.get("hand")
    hand_cards = [_card_value_dict(c) for c in _cards_of(hand_area)]

    jokers = [{"key": _attr(j, "center_key") or _attr(j, "key"),
               "label": _attr(j, "label"),
               "modifier": {"rental": bool((_attr(j, "modifier") or {}).get("rental")
                                           if isinstance(_attr(j, "modifier"), dict)
                                           else getattr(_attr(j, "modifier"), "rental", False)),
                             "perishable": bool((_attr(j, "modifier") or {}).get("perishable")
                                                if isinstance(_attr(j, "modifier"), dict)
                                                else getattr(_attr(j, "modifier"), "perishable", False)),
                             "eternal": bool((_attr(j, "modifier") or {}).get("eternal")
                                             if isinstance(_attr(j, "modifier"), dict)
                                             else getattr(_attr(j, "modifier"), "eternal", False))}}
              for j in _cards_of(raw.get("jokers"))]

    consumables = [{"set": str(_attr(_attr(c, "ability"), "set") or _attr(c, "set") or ""),
                    "label": _attr(c, "label")}
                   for c in _cards_of(raw.get("consumables"))]

    shop_cards = [{"cost": {"buy": float((_attr(c, "cost") or {}).get("buy", 999)
                                         if isinstance(_attr(c, "cost"), dict)
                                         else getattr(_attr(c, "cost"), "buy", 999)),
                   },
                   "set": str(_attr(_attr(c, "ability"), "set") or _attr(c, "set") or ""),
                   "label": _attr(c, "label")}
                  for c in _cards_of(raw.get("shop"))]

    pack_cards = [{"label": _attr(c, "label")} for c in _cards_of(raw.get("pack"))]

    blind = blind_view(raw)

    return {
        "state": phase,
        "ante_num": int((rr.get("ante")) or 1),
        "money": float(raw.get("dollars", 0) or 0),
        "blinds": {"boss": blind} if blind else {},
        "blind": blind,
        "hand": {"cards": hand_cards},
        "hands": _hands_view(raw.get("hand_levels")),
        "round": {"hands_left": int(_attr(cur, "hands_left", 1) or 0),
                  "discards_left": int(_attr(cur, "discards_left", 0) or 0),
                  "chips": float(raw.get("chips", 0) or 0)},
        "jokers": {"cards": jokers, "count": len(jokers),
                   "limit": int(_attr(_attr(raw, "jokers"), "limit", 5) or 5)},
        "consumables": {"cards": consumables,
                        "count": len(consumables),
                        "limit": 2},
        "shop": {"cards": shop_cards},
        "pack": {"cards": pack_cards},
    }
