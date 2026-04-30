"""
╔══════════════════════════════════════════════════════════════════════╗
║   OPTIMIZED MATCHMAKING SYSTEM  v4.0  — PRODUCTION                   ║
║   Navigation : 10-page sidebar                                       ║
║   Storage    : JSON (matchmaking_data.json)                          ║
║   Structures : AVL Tree · Max Heap (leaderboard) · Min Heap (queue) ║
║   Rating     : ELO with adaptive K-factor                           ║
║   NEW        : Player Profiles · Team Matchmaking · Tournament      ║
╚══════════════════════════════════════════════════════════════════════╝

VERSION 4.0 FEATURES
────────────────────
 ✔  Player Profiles (avatar, bio, stats page)
 ✔  Team Matchmaking (2v2, 3v3, 4v4 modes)
 ✔  Tournament Mode (create/join brackets, single/double elimination)

DATA STRUCTURES
───────────────
 AVLTree        – global player store, keyed (rating, name)
                  insert/delete/update  O(log n)
                  range_query           O(log n + k)
 MaxHeapManager – leaderboard (highest rating first)
                  top_k                 O(k log n)
 MinHeapManager – per-mode queue cheapest-wait player first
                  find_lowest           O(log n)
                  used for "lowest rated first" queue dispatch
"""

import streamlit as st
import heapq
import time
import random
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE  = "matchmaking_data.json"
MODES      = ["Ranked", "Casual", "Turbo"]
THRESHOLD  = 150
K_PROV     = 40
K_EST      = 20
K_HIGH     = 10

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Player:
    name:          str
    rating:        float
    mode:          str
    wins:          int   = 0
    losses:        int   = 0
    draws:         int   = 0
    in_queue:      bool  = False
    registered_at: float = field(default_factory=time.time)   # FIX: added field

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games_played * 100) if self.games_played else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Player":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class Match:
    match_id:       int
    player1:        str
    player2:        str
    mode:           str
    rating1_before: float
    rating2_before: float
    result:         Optional[str]   = None
    rating1_after:  Optional[float] = None
    rating2_after:  Optional[float] = None
    timestamp:      float           = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Match":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: PLAYER PROFILE DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerProfile:
    player_name:    str
    avatar:         str = "🎮"
    bio:            str = ""
    favorite_mode: str = "Ranked"
    country:        str = ""
    discord:        str = ""
    created_at:     float = field(default_factory=time.time)
    last_active:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerProfile":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: TEAM DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Team:
    team_id:        int
    name:           str
    captain:        str
    members:        List[str] = field(default_factory=list)
    tag:            str = ""
    wins:           int = 0
    losses:         int = 0
    created_at:     float = field(default_factory=time.time)

    @property
    def games_played(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games_played * 100) if self.games_played else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Team":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: TOURNAMENT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tournament:
    tournament_id:  int
    name:          str
    mode:          str
    format:        str = "single"  # single, double
    max_teams:     int = 8
    teams:         List[str] = field(default_factory=list)  # team names
    matches:       List[dict] = field(default_factory=list)  # tournament matches
    status:        str = "open"  # open, in_progress, completed
    prize:         str = ""
    created_by:    str = ""
    created_at:    float = field(default_factory=time.time)
    started_at:    Optional[float] = None
    ended_at:      Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Tournament":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ─────────────────────────────────────────────────────────────────────────────
# AVL TREE
# ─────────────────────────────────────────────────────────────────────────────

class AVLNode:
    """Node keyed on (rating, name) — unique even with duplicate ratings."""
    __slots__ = ("player", "key", "left", "right", "height")

    def __init__(self, player: "Player"):
        self.player = player
        self.key: Tuple[float, str] = (player.rating, player.name)
        self.left:   Optional["AVLNode"] = None
        self.right:  Optional["AVLNode"] = None
        self.height: int = 1


class AVLTree:
    """
    Self-balancing BST.
    insert / delete / update_rating   O(log n)
    search (by name)                  O(1)   via secondary dict
    range_query(lo, hi)               O(log n + k)
    find_closest(target)              O(log n)
    in_order()                        O(n)
    """

    def __init__(self):
        self.root: Optional[AVLNode] = None
        self.size: int = 0
        self._idx: dict = {}

    @staticmethod
    def _h(n) -> int:
        return n.height if n else 0

    @staticmethod
    def _bf(n) -> int:
        return AVLTree._h(n.left) - AVLTree._h(n.right)

    @staticmethod
    def _uh(n):
        n.height = 1 + max(AVLTree._h(n.left), AVLTree._h(n.right))

    @staticmethod
    def _rr(y) -> "AVLNode":
        x = y.left
        y.left = x.right
        x.right = y
        AVLTree._uh(y)
        AVLTree._uh(x)
        return x

    @staticmethod
    def _rl(x) -> "AVLNode":
        y = x.right
        x.right = y.left
        y.left = x
        AVLTree._uh(x)
        AVLTree._uh(y)
        return y

    @staticmethod
    def _bal(n) -> "AVLNode":
        AVLTree._uh(n)
        bf = AVLTree._bf(n)
        if bf > 1:
            if AVLTree._bf(n.left) < 0:
                n.left = AVLTree._rl(n.left)
            return AVLTree._rr(n)
        if bf < -1:
            if AVLTree._bf(n.right) > 0:
                n.right = AVLTree._rr(n.right)
            return AVLTree._rl(n)
        return n

    def insert(self, player: "Player"):
        if player.name in self._idx:
            return
        self.root = self._ins(self.root, player)
        self._idx[player.name] = player
        self.size += 1

    def delete(self, player: "Player"):
        if player.name not in self._idx:
            return
        self.root = self._del(self.root, (player.rating, player.name))
        del self._idx[player.name]
        self.size -= 1

    def update_rating(self, player: "Player", new_rating: float):
        self.delete(player)
        player.rating = new_rating
        self.insert(player)

    def search(self, name: str):
        return self._idx.get(name)

    def range_query(self, lo: float, hi: float, mode=None) -> list:
        res = []
        self._rq(self.root, lo, hi, mode, res)
        return res

    def find_closest(self, target: float, mode=None, exclude=None):
        best = [None, float("inf")]

        def _s(n):
            if not n:
                return
            p = n.player
            if p.name != exclude and (mode is None or p.mode == mode):
                d = abs(p.rating - target)
                if d < best[1]:
                    best[0], best[1] = p, d
            if target < n.key[0]:
                _s(n.left)
                if n.key[0] - target < best[1]:
                    _s(n.right)
            else:
                _s(n.right)
                if target - n.key[0] < best[1]:
                    _s(n.left)

        _s(self.root)
        return best[0]

    def in_order(self) -> list:
        res = []
        self._io(self.root, res)
        return res

    def _ins(self, n, p):
        if not n:
            return AVLNode(p)
        k = (p.rating, p.name)
        if k < n.key:
            n.left = self._ins(n.left, p)
        elif k > n.key:
            n.right = self._ins(n.right, p)
        return self._bal(n)

    def _del(self, n, key):
        if not n:
            return None
        if key < n.key:
            n.left = self._del(n.left, key)
        elif key > n.key:
            n.right = self._del(n.right, key)
        else:
            if not n.left:  return n.right
            if not n.right: return n.left
            s = self._min(n.right)
            n.player = s.player
            n.key    = s.key
            n.right  = self._del(n.right, s.key)
        return self._bal(n)

    @staticmethod
    def _min(n):
        while n.left:
            n = n.left
        return n

    def _io(self, n, res):
        if not n: return
        self._io(n.left, res)
        res.append(n.player)
        self._io(n.right, res)

    def _rq(self, n, lo, hi, mode, res):
        if not n: return
        r = n.player.rating
        if lo <= r:
            self._rq(n.left, lo, hi, mode, res)
        if lo <= r <= hi and (mode is None or n.player.mode == mode):
            res.append(n.player)
        if r <= hi:
            self._rq(n.right, lo, hi, mode, res)


# ─────────────────────────────────────────────────────────────────────────────
# MAX HEAP MANAGER  — leaderboard (highest rating first)
# ─────────────────────────────────────────────────────────────────────────────

class MaxHeapManager:
    """
    Max-heap via negated values in Python heapq.  Lazy deletion.
    push / update  O(log n)
    top_k          O(k log n)
    """

    def __init__(self):
        self._heap:  list = []   # (-rating, name)
        self._valid: dict = {}   # name -> current rating

    def push(self, player: Player):
        self._valid[player.name] = player.rating
        heapq.heappush(self._heap, (-player.rating, player.name))

    def update(self, player: Player, new_rating: float):
        self._valid[player.name] = new_rating
        heapq.heappush(self._heap, (-new_rating, player.name))

    def remove(self, name: str):
        self._valid.pop(name, None)

    def _ok(self, neg_r: float, name: str) -> bool:
        return name in self._valid and -neg_r == self._valid[name]

    def top_k(self, k: int, all_players: dict) -> list:
        tmp  = list(self._heap)
        out  = []
        seen = set()
        while tmp and len(out) < k:
            neg_r, name = heapq.heappop(tmp)
            if name in seen: continue
            seen.add(name)
            if not self._ok(neg_r, name): continue
            if name in all_players:
                out.append(all_players[name])
        return out

    def peek(self, all_players: dict):
        tmp = list(self._heap)
        while tmp:
            neg_r, name = heapq.heappop(tmp)
            if self._ok(neg_r, name) and name in all_players:
                return all_players[name]
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MIN HEAP MANAGER  — per-mode queue (lowest rating first)
# ─────────────────────────────────────────────────────────────────────────────

class MinHeapManager:
    """
    Min-heap on player ratings — dispatches the lowest-rated queued player
    first, ensuring underdogs get matched fairly.

    enqueue(player)          O(log n)
    dequeue_min(players)     O(log n)
    peek_min(players)        O(log n)
    remove(name)             O(1)  lazy
    size                     O(1)
    all_queued(players)      O(n log n)
    """

    def __init__(self):
        self._heap:       list = []   # (rating, name)
        self._valid:      dict = {}   # name -> rating
        self._entry_time: dict = {}   # name -> enqueue timestamp

    def enqueue(self, player: Player):
        self._valid[player.name]      = player.rating
        self._entry_time[player.name] = time.time()
        heapq.heappush(self._heap, (player.rating, player.name))

    def dequeue_min(self, all_players: dict):
        """Pop and return the lowest-rated valid queued player.  O(log n)."""
        while self._heap:
            r, name = heapq.heappop(self._heap)
            if name in self._valid and r == self._valid[name] and name in all_players:
                del self._valid[name]
                self._entry_time.pop(name, None)
                return all_players[name]
        return None

    def peek_min(self, all_players: dict):
        """Lowest-rated valid queued player without removing.  O(log n)."""
        tmp = list(self._heap)
        while tmp:
            r, name = heapq.heappop(tmp)
            if name in self._valid and r == self._valid[name] and name in all_players:
                return all_players[name]
        return None

    def remove(self, name: str):
        """Lazy invalidation.  O(1)."""
        self._valid.pop(name, None)
        self._entry_time.pop(name, None)

    @property
    def size(self) -> int:
        return len(self._valid)

    def all_queued(self, all_players: dict) -> list:
        """All valid queued (Player, wait_seconds) sorted by rating asc.  O(n log n)."""
        now    = time.time()
        result = []
        for name, rating in self._valid.items():
            if name in all_players:
                wait = now - self._entry_time.get(name, now)
                result.append((all_players[name], wait))
        result.sort(key=lambda x: x[0].rating)
        return result

    def wait_time(self, name: str) -> float:
        if name not in self._entry_time:
            return 0.0
        return time.time() - self._entry_time[name]


# ─────────────────────────────────────────────────────────────────────────────
# ELO RATING
# ─────────────────────────────────────────────────────────────────────────────

class EloRating:
    @staticmethod
    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    @staticmethod
    def k_factor(p: Player) -> int:
        if p.rating > 2400:      return K_HIGH
        if p.games_played < 10:  return K_PROV
        return K_EST

    @classmethod
    def calculate(cls, p1: Player, p2: Player, result: str) -> Tuple[float, float]:
        e1 = cls.expected(p1.rating, p2.rating)
        e2 = 1.0 - e1                            # FIX: explicit e2
        s1, s2 = {"player1": (1.0, 0.0), "player2": (0.0, 1.0), "draw": (0.5, 0.5)}[result]
        k1, k2 = cls.k_factor(p1), cls.k_factor(p2)
        return (round(p1.rating + k1 * (s1 - e1), 2),
                round(p2.rating + k2 * (s2 - e2), 2))   # FIX: s2 - e2


# ─────────────────────────────────────────────────────────────────────────────
# JSON PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_data(players: dict, matches: list, counter: int, profiles: dict = None, 
              teams: dict = None, tournaments: list = None, team_counter: int = 0,
              tournament_counter: int = 0):
    with open(DATA_FILE, "w") as f:
        json.dump({
            "match_counter": counter,
            "players": [p.to_dict() for p in players.values()],
            "matches":  [m.to_dict() for m in matches],
            "profiles": {k: v.to_dict() for k, v in (profiles or {}).items()},
            "teams": {k: v.to_dict() for k, v in (teams or {}).items()},
            "team_counter": team_counter,
            "tournaments": [t.to_dict() for t in (tournaments or [])],
            "tournament_counter": tournament_counter,
        }, f, indent=2)


def load_data() -> Tuple[dict, list, int, dict, dict, list, int, int]:
    if not os.path.exists(DATA_FILE):
        return {}, [], 0, {}, {}, [], 0, 0
    try:
        with open(DATA_FILE) as f:
            raw = json.load(f)
        players = {}
        for d in raw.get("players", []):
            p = Player.from_dict(d)
            p.in_queue = False
            players[p.name] = p
        matches = [Match.from_dict(d) for d in raw.get("matches", [])]
        
        # Load new data structures
        profiles = {k: PlayerProfile.from_dict(v) for k, v in raw.get("profiles", {}).items()}
        teams = {k: Team.from_dict(v) for k, v in raw.get("teams", {}).items()}
        tournaments = [Tournament.from_dict(d) for d in raw.get("tournaments", [])]
        
        return (players, matches, raw.get("match_counter", 0), 
                profiles, teams, tournaments,
                raw.get("team_counter", 0), raw.get("tournament_counter", 0))
    except Exception:
        return {}, [], 0, {}, {}, [], 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# MATCHMAKING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class MatchmakingSystem:
    """
    Core engine.
        _players   dict[name -> Player]           O(1) lookup
        _avl       AVLTree                         sorted by rating
        _max_heap  MaxHeapManager                  leaderboard
        _min_heaps dict[mode -> MinHeapManager]    per-mode queue dispatch
        _avl_q     dict[mode -> AVLTree]           per-mode queue range queries
        _matches   List[Match]
        _profiles  dict[name -> PlayerProfile]    player profiles
        _teams     dict[name -> Team]              team storage
        _tournaments List[Tournament]              tournament storage
    Every mutation flushes to JSON immediately.
    """

    def __init__(self):
        self._players:   dict = {}
        self._avl        = AVLTree()
        self._max_heap   = MaxHeapManager()
        self._min_heaps  = {m: MinHeapManager() for m in MODES}
        self._avl_q      = {m: AVLTree()        for m in MODES}
        self._matches:   list = []
        self._counter:   int  = 0
        # New: profiles, teams, tournaments
        self._profiles:  dict = {}
        self._teams:     dict = {}
        self._tournaments: list = []
        self._team_counter: int = 0
        self._tournament_counter: int = 0
        self._load()

    def _load(self):
        (players, matches, counter, profiles, teams, tournaments, 
         team_counter, tournament_counter) = load_data()
        self._players  = players
        self._matches  = matches
        self._counter  = counter
        self._profiles = profiles
        self._teams    = teams
        self._tournaments = tournaments
        self._team_counter = team_counter
        self._tournament_counter = tournament_counter
        for p in self._players.values():
            self._avl.insert(p)
            self._max_heap.push(p)

    def _save(self):
        save_data(self._players, self._matches, self._counter, 
                  self._profiles, self._teams, self._tournaments,
                  self._team_counter, self._tournament_counter)

    # ── registration ──────────────────────────────────────────────────────────

    def register_player(self, name: str, rating: float, mode: str) -> Tuple[bool, str]:
        name = name.strip()
        if not name:            return False, "Player name cannot be empty."
        if name in self._players: return False, f"'{name}' is already registered."
        if mode not in MODES:   return False, f"Unknown mode '{mode}'."
        if not (100 <= rating <= 3000): return False, "Rating must be between 100 and 3000."
        p = Player(name=name, rating=rating, mode=mode)
        self._players[name] = p
        self._avl.insert(p)
        self._max_heap.push(p)
        # Auto-join queue on registration
        p.in_queue = True
        self._avl_q[p.mode].insert(p)
        self._min_heaps[p.mode].enqueue(p)
        self._save()
        return True, f"'{name}' registered at {rating:.0f} ({mode}) and joined queue."

    def delete_player(self, name: str) -> Tuple[bool, str]:
        p = self._players.get(name)
        if not p: return False, f"Player '{name}' not found."
        if p.in_queue:
            self._avl_q[p.mode].delete(p)
            self._min_heaps[p.mode].remove(p.name)
        self._avl.delete(p)
        self._max_heap.remove(name)
        del self._players[name]
        self._save()
        return True, f"'{name}' deleted."

    # ── queue ─────────────────────────────────────────────────────────────────

    def join_queue(self, name: str) -> Tuple[bool, str]:
        p = self._players.get(name)
        if not p:       return False, f"Player '{name}' not found."
        if p.in_queue:  return False, f"'{name}' is already in the queue."
        p.in_queue = True
        self._avl_q[p.mode].insert(p)
        self._min_heaps[p.mode].enqueue(p)
        self._save()
        return True, f"'{name}' joined the {p.mode} queue."

    def leave_queue(self, name: str) -> Tuple[bool, str]:
        p = self._players.get(name)
        if not p:          return False, f"Player '{name}' not found."
        if not p.in_queue: return False, f"'{name}' is not in the queue."
        p.in_queue = False
        self._avl_q[p.mode].delete(p)
        self._min_heaps[p.mode].remove(p.name)
        self._save()
        return True, f"'{name}' left the queue."

    def queue_status(self, mode: str) -> list:
        return self._avl_q[mode].in_order()

    def queue_with_wait(self, mode: str) -> list:
        return self._min_heaps[mode].all_queued(self._players)

    def lowest_rated_in_queue(self, mode: str):
        return self._min_heaps[mode].peek_min(self._players)

    # ── matchmaking ───────────────────────────────────────────────────────────

    def find_match(self, name: str) -> Tuple[bool, str]:
        p = self._players.get(name)
        if not p:          return False, f"Player '{name}' not found."
        if not p.in_queue: return False, f"'{name}' must join the queue first."
        tree  = self._avl_q[p.mode]
        cands = [c for c in
                 tree.range_query(p.rating - THRESHOLD, p.rating + THRESHOLD, p.mode)
                 if c.name != name]
        if not cands:
            opp = tree.find_closest(p.rating, mode=p.mode, exclude=name)
            if not opp:
                return False, f"No opponents available in the {p.mode} queue."
        else:
            opp = min(cands, key=lambda c: abs(c.rating - p.rating))
        return self._create_match(p, opp)

    def create_match_manual(self, n1: str, n2: Optional[str] = None) -> Tuple[bool, str]:
        """
        Auto-match: If only n1 is provided, find best opponent for that player.
        If both n1 and n2 are provided, create match between them directly.
        """
        p1 = self._players.get(n1)
        if not p1: return False, f"Player '{n1}' not found."
        
        # If n2 is provided, create direct match
        if n2:
            p2 = self._players.get(n2)
            if not p2: return False, f"Player '{n2}' not found."
            if n1 == n2: return False, "Cannot match a player against themselves."
            for p in [p1, p2]:
                if not p.in_queue:
                    p.in_queue = True
                    self._avl_q[p.mode].insert(p)
                    self._min_heaps[p.mode].enqueue(p)
            return self._create_match(p1, p2)
        
        # Auto-match: find best opponent for p1
        if not p1.in_queue:
            p1.in_queue = True
            self._avl_q[p1.mode].insert(p1)
            self._min_heaps[p1.mode].enqueue(p1)
        
        tree = self._avl_q[p1.mode]
        cands = [c for c in tree.range_query(p1.rating - THRESHOLD, p1.rating + THRESHOLD, p1.mode) if c.name != n1]
        
        if not cands:
            opp = tree.find_closest(p1.rating, mode=p1.mode, exclude=n1)
            if not opp:
                return False, f"No opponents available in the {p1.mode} queue."
        else:
            opp = min(cands, key=lambda c: abs(c.rating - p1.rating))
        
        return self._create_match(p1, opp)

    def _create_match(self, p1: Player, p2: Player) -> Tuple[bool, str]:
        self._counter += 1
        m = Match(match_id=self._counter, player1=p1.name, player2=p2.name,
                  mode=p1.mode, rating1_before=p1.rating, rating2_before=p2.rating)
        self._matches.append(m)
        for p in [p1, p2]:
            p.in_queue = False
            self._avl_q[p.mode].delete(p)
            self._min_heaps[p.mode].remove(p.name)
        self._save()
        return True, (f"Match #{m.match_id}: {p1.name} ({p1.rating:.0f}) vs "
                      f"{p2.name} ({p2.rating:.0f}) — {p1.mode}")

    # ── resolve ───────────────────────────────────────────────────────────────

    def resolve_match(self, match_id: int, result: str) -> Tuple[bool, str]:
        m = next((x for x in self._matches if x.match_id == match_id), None)
        if not m:      return False, f"Match #{match_id} not found."
        if m.result:   return False, f"Match #{match_id} already resolved."
        p1 = self._players.get(m.player1)
        p2 = self._players.get(m.player2)
        if not p1 or not p2: return False, "One or both players not found."
        r1, r2 = EloRating.calculate(p1, p2, result)
        if   result == "player1": p1.wins+=1;  p2.losses+=1
        elif result == "player2": p2.wins+=1;  p1.losses+=1
        else:                     p1.draws+=1; p2.draws+=1
        self._avl.update_rating(p1, r1)
        self._avl.update_rating(p2, r2)
        self._max_heap.update(p1, r1)
        self._max_heap.update(p2, r2)
        m.result = result; m.rating1_after = r1; m.rating2_after = r2
        self._save()
        d1, d2 = r1 - m.rating1_before, r2 - m.rating2_before
        return True, (f"Match #{match_id} resolved | "
                      f"{p1.name}: {m.rating1_before:.0f}->{r1:.0f} ({d1:+.0f}) | "
                      f"{p2.name}: {m.rating2_before:.0f}->{r2:.0f} ({d2:+.0f})")

    # ── queries ───────────────────────────────────────────────────────────────

    def leaderboard(self, k: int = 10) -> list:
        return self._max_heap.top_k(k, self._players)

    def all_players_sorted(self) -> list:
        return list(reversed(self._avl.in_order()))

    def pending_matches(self)   -> list:
        return [m for m in self._matches if m.result is None]

    def completed_matches(self) -> list:
        return [m for m in self._matches if m.result is not None]

    def player_names(self) -> list:
        return sorted(self._players.keys())

    def get_player(self, name: str):
        return self._players.get(name)

    # ── Player Profile Methods ───────────────────────────────────────────────

    def create_profile(self, player_name: str, avatar: str = "🎮", bio: str = "",
                       favorite_mode: str = "Ranked", country: str = "",
                       discord: str = "") -> Tuple[bool, str]:
        if player_name not in self._players:
            return False, f"Player '{player_name}' not found."
        if player_name in self._profiles:
            return False, f"Profile already exists for '{player_name}'."
        profile = PlayerProfile(
            player_name=player_name,
            avatar=avatar,
            bio=bio,
            favorite_mode=favorite_mode,
            country=country,
            discord=discord
        )
        self._profiles[player_name] = profile
        self._save()
        return True, f"Profile created for '{player_name}'."

    def update_profile(self, player_name: str, **kwargs) -> Tuple[bool, str]:
        if player_name not in self._profiles:
            return False, f"Profile not found for '{player_name}'."
        profile = self._profiles[player_name]
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.last_active = time.time()
        self._save()
        return True, f"Profile updated for '{player_name}'."

    def get_profile(self, player_name: str):
        return self._profiles.get(player_name)

    # ── Team Methods ─────────────────────────────────────────────────────────

    def create_team(self, name: str, captain: str, tag: str = "") -> Tuple[bool, str]:
        if name in self._teams:
            return False, f"Team '{name}' already exists."
        if captain not in self._players:
            return False, f"Captain '{captain}' not found as a player."
        self._team_counter += 1
        team = Team(
            team_id=self._team_counter,
            name=name,
            captain=captain,
            members=[captain],
            tag=tag
        )
        self._teams[name] = team
        self._save()
        return True, f"Team '{name}' created with captain {captain}."

    def join_team(self, team_name: str, player_name: str) -> Tuple[bool, str]:
        team = self._teams.get(team_name)
        if not team:
            return False, f"Team '{team_name}' not found."
        if player_name in team.members:
            return False, f"'{player_name}' is already in the team."
        if player_name not in self._players:
            return False, f"Player '{player_name}' not found."
        team.members.append(player_name)
        self._save()
        return True, f"'{player_name}' joined team '{team_name}'."

    def leave_team(self, team_name: str, player_name: str) -> Tuple[bool, str]:
        team = self._teams.get(team_name)
        if not team:
            return False, f"Team '{team_name}' not found."
        if player_name not in team.members:
            return False, f"'{player_name}' is not in the team."
        if player_name == team.captain:
            return False, "Captain cannot leave the team. Transfer ownership first."
        team.members.remove(player_name)
        self._save()
        return True, f"'{player_name}' left team '{team_name}'."

    def get_team(self, team_name: str):
        return self._teams.get(team_name)

    def team_names(self) -> list:
        return sorted(self._teams.keys())

    # ── Tournament Methods ───────────────────────────────────────────────────

    def create_tournament(self, name: str, mode: str, format: str = "single",
                          max_teams: int = 8, prize: str = "", created_by: str = "") -> Tuple[bool, str]:
        if mode not in MODES:
            return False, f"Unknown mode '{mode}'."
        self._tournament_counter += 1
        tournament = Tournament(
            tournament_id=self._tournament_counter,
            name=name,
            mode=mode,
            format=format,
            max_teams=max_teams,
            prize=prize,
            created_by=created_by
        )
        self._tournaments.append(tournament)
        self._save()
        return True, f"Tournament '{name}' created ({format} elimination, {max_teams} teams)."

    def join_tournament(self, tournament_id: int, team_name: str) -> Tuple[bool, str]:
        tournament = next((t for t in self._tournaments if t.tournament_id == tournament_id), None)
        if not tournament:
            return False, f"Tournament #{tournament_id} not found."
        if tournament.status != "open":
            return False, f"Tournament is {tournament.status}."
        if team_name not in self._teams:
            return False, f"Team '{team_name}' not found."
        if team_name in tournament.teams:
            return False, f"Team '{team_name}' already in tournament."
        if len(tournament.teams) >= tournament.max_teams:
            return False, "Tournament is full."
        tournament.teams.append(team_name)
        self._save()
        return True, f"Team '{team_name}' joined tournament '{tournament.name}'."

    def start_tournament(self, tournament_id: int) -> Tuple[bool, str]:
        tournament = next((t for t in self._tournaments if t.tournament_id == tournament_id), None)
        if not tournament:
            return False, f"Tournament #{tournament_id} not found."
        if len(tournament.teams) < 2:
            return False, "Need at least 2 teams to start."
        if tournament.status != "open":
            return False, f"Tournament is {tournament.status}."
        
        # Generate bracket based on format
        tournament.status = "in_progress"
        tournament.started_at = time.time()
        
        # Simple single elimination - shuffle and pair
        teams = list(tournament.teams)
        random.shuffle(teams)
        tournament.matches = []
        
        # Create first round matches
        for i in range(0, len(teams) - 1, 2):
            match = {
                "round": 1,
                "team1": teams[i],
                "team2": teams[i + 1],
                "winner": None,
                "score1": 0,
                "score2": 0
            }
            tournament.matches.append(match)
        
        # Byes if odd number
        if len(teams) % 2 == 1:
            bye_team = teams[-1]
            tournament.matches.append({
                "round": 1,
                "team1": bye_team,
                "team2": "BYE",
                "winner": bye_team,
                "score1": 1,
                "score2": 0
            })
        
        self._save()
        return True, f"Tournament '{tournament.name}' started!"

    def resolve_tournament_match(self, tournament_id: int, round_num: int, 
                                   team1: str, team2: str, winner: str) -> Tuple[bool, str]:
        tournament = next((t for t in self._tournaments if t.tournament_id == tournament_id), None)
        if not tournament:
            return False, f"Tournament #{tournament_id} not found."
        
        for match in tournament.matches:
            if match["round"] == round_num and match["team1"] == team1 and match["team2"] == team2:
                match["winner"] = winner
                # Update scores
                if winner == team1:
                    match["score1"] = 1
                else:
                    match["score2"] = 1
                break
        
        # Check if tournament is complete
        all_resolved = all(m["winner"] is not None for m in tournament.matches if m["team2"] != "BYE")
        if all_resolved and len(tournament.matches) > 0:
            # Simple winner determination
            final_match = [m for m in tournament.matches if m["round"] == max(m["round"] for m in tournament.matches)]
            if final_match and final_match[0]["winner"]:
                tournament.status = "completed"
                tournament.ended_at = time.time()
        
        self._save()
        return True, f"Match result recorded. Winner: {winner}"

    def get_tournament(self, tournament_id: int):
        return next((t for t in self._tournaments if t.tournament_id == tournament_id), None)

    def list_tournaments(self, status: str = None) -> list:
        if status:
            return [t for t in self._tournaments if t.status == status]
        return self._tournaments


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TIER_CONFIG = {
    "Grandmaster": {"min": 2400, "color": "#FFD700", "icon": "👑"},
    "Diamond":     {"min": 2000, "color": "#B9F2FF", "icon": "💎"},
    "Gold":        {"min": 1600, "color": "#FFC857", "icon": "🥇"},
    "Silver":      {"min": 1200, "color": "#C0C0C0", "icon": "🥈"},
    "Bronze":      {"min":    0, "color": "#CD7F32", "icon": "🥉"},
}

def get_tier(rating: float) -> Tuple[str, str, str]:
    for name, cfg in TIER_CONFIG.items():
        if rating >= cfg["min"]:
            return name, cfg["color"], cfg["icon"]
    return "Bronze", "#CD7F32", "🥉"

def mode_color(mode: str) -> str:
    return {"Ranked": "#ef4444", "Casual": "#3b82f6", "Turbo": "#a855f7"}.get(mode, "#888")

def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y  %H:%M")

def fmt_wait(sec: float) -> str:
    if sec < 60:   return f"{sec:.0f}s"
    if sec < 3600: return f"{sec/60:.0f}m {sec%60:.0f}s"
    return f"{sec/3600:.1f}h"

def _page_key(page_str: str) -> str:
    """FIX: regex-based extraction, immune to emoji variation selectors (U+FE0F)."""
    return re.sub(r"^[^\w]+", "", page_str).strip()


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg,#0a0a0f 0%,#0f0f1a 55%,#090f0a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0d1a,#0a0a14) !important;
    border-right: 1px solid #1e1e3a !important;
}
section[data-testid="stSidebar"] * { color: #c4c9e0 !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 3px; }
section[data-testid="stSidebar"] .stRadio label {
    border: 1px solid transparent; border-radius: 8px;
    padding: 9px 14px !important; font-size: 0.88rem; font-weight: 500;
    cursor: pointer; transition: all 0.18s; color: #9da3bf !important; width: 100%;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
}

/* ── Titles ── */
.page-title {
    font-family:'Rajdhani',sans-serif; font-size:2.1rem; font-weight:700;
    letter-spacing:0.03em;
    background:linear-gradient(90deg,#6366f1,#a78bfa,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-bottom:0.15rem;
}
.page-sub { color:#64748b; font-size:0.8rem; margin-bottom:1.3rem; text-transform:uppercase; letter-spacing:0.05em; }
.sidebar-brand { font-family:'Rajdhani',sans-serif; font-size:1.3rem; font-weight:700; color:#a78bfa !important; letter-spacing:0.05em; }
.sidebar-sub { font-size:0.66rem; color:#475569 !important; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.3rem; }

/* ── Cards ── */
.metric-card { background:rgba(255,255,255,0.03); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px; text-align:center; transition:border-color 0.2s; }
.metric-card:hover { border-color:rgba(99,102,241,0.5); }
.metric-val { font-family:'Rajdhani',sans-serif; font-size:2.1rem; font-weight:700; color:#a78bfa; line-height:1; }
.metric-lbl { font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-top:3px; }

.player-row { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px 16px; margin-bottom:7px; display:flex; align-items:center; transition:all 0.18s; }
.player-row:hover { background:rgba(99,102,241,0.06); border-color:rgba(99,102,241,0.25); }
.podium-1 { background:linear-gradient(135deg,rgba(255,215,0,.12),rgba(255,215,0,.03)); border-color:rgba(255,215,0,.35)!important; }
.podium-2 { background:linear-gradient(135deg,rgba(192,192,192,.10),rgba(192,192,192,.03)); border-color:rgba(192,192,192,.25)!important; }
.podium-3 { background:linear-gradient(135deg,rgba(205,127,50,.10),rgba(205,127,50,.03)); border-color:rgba(205,127,50,.25)!important; }

.match-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:18px; margin-bottom:10px; }
.match-vs { font-family:'Rajdhani',sans-serif; font-size:1.6rem; font-weight:700; color:#ef4444; text-align:center; }

/* ── Badges ── */
.badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:0.68rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; }
.badge-ranked  { background:rgba(239,68,68,.15);  color:#ef4444; border:1px solid rgba(239,68,68,.3); }
.badge-casual  { background:rgba(59,130,246,.15); color:#3b82f6; border:1px solid rgba(59,130,246,.3); }
.badge-turbo   { background:rgba(168,85,247,.15); color:#a855f7; border:1px solid rgba(168,85,247,.3); }
.q-pill   { display:inline-block; background:rgba(168,85,247,.15); color:#a855f7; border:1px solid rgba(168,85,247,.3); border-radius:12px; padding:1px 7px; font-size:0.67rem; font-weight:600; margin-left:5px; }
.min-pill { display:inline-block; background:rgba(56,189,248,.15); color:#38bdf8; border:1px solid rgba(56,189,248,.3); border-radius:12px; padding:1px 7px; font-size:0.67rem; font-weight:600; margin-left:5px; }

/* ── Inputs & buttons ── */
.stTextInput input, .stNumberInput input {
    background:rgba(255,255,255,0.04)!important; border:1px solid rgba(255,255,255,0.1)!important;
    border-radius:8px!important; color:#e2e8f0!important;
}
.stButton > button {
    background:linear-gradient(135deg,#6366f1,#4f46e5)!important; color:#fff!important;
    border:none!important; border-radius:8px!important; font-weight:600!important;
    padding:9px 18px!important; transition:all 0.18s!important;
}
.stButton > button:hover { background:linear-gradient(135deg,#7c7ff5,#6366f1)!important; transform:translateY(-1px); box-shadow:0 4px 14px rgba(99,102,241,0.4)!important; }

/* ── Alerts ── */
div[data-testid="stSuccess"] { background:rgba(34,197,94,.08)!important;  border-color:rgba(34,197,94,.3)!important; }
div[data-testid="stError"]   { background:rgba(239,68,68,.08)!important;  border-color:rgba(239,68,68,.3)!important; }
div[data-testid="stInfo"]    { background:rgba(99,102,241,.08)!important; border-color:rgba(99,102,241,.3)!important; }
div[data-testid="stWarning"] { background:rgba(251,191,36,.08)!important; border-color:rgba(251,191,36,.3)!important; }

hr { border-color:rgba(255,255,255,0.07)!important; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#2d2d4a; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────────────────────────────────────────────

def get_sys() -> MatchmakingSystem:
    if "sys" not in st.session_state:
        st.session_state["sys"] = MatchmakingSystem()
    if "activity" not in st.session_state:
        st.session_state["activity"] = []
    return st.session_state["sys"]

def add_activity(msg: str, kind: str = "info"):
    icons = {"success":"✅","error":"❌","info":"ℹ️","warn":"⚠️","match":"⚔️"}
    st.session_state.setdefault("activity",[]).insert(0,{"msg":msg,"kind":kind,"ts":time.time()})
    st.session_state["activity"] = st.session_state["activity"][:30]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ① — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">⚔ Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">System overview · live stats</div>', unsafe_allow_html=True)

    players = ms.all_players_sorted()
    pending = ms.pending_matches()
    done    = ms.completed_matches()
    # FIX: use mode_nm to avoid shadowing columns result with 'col'
    q_total = sum(ms._min_heaps[mode_nm].size for mode_nm in MODES)
    top1    = ms.leaderboard(1)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, val, lbl in [
        (mc1, len(players),  "Total Players"),
        (mc2, q_total,       "In Queue"),
        (mc3, len(pending),  "Live Matches"),
        (mc4, len(done),     "Completed"),
        (mc5, f"{top1[0].rating:.0f}" if top1 else "—", "Top Rating"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### 🏆 Top Players <span style='font-size:0.73rem;color:#64748b'>(Max Heap)</span>",
                    unsafe_allow_html=True)
        podium = ms.leaderboard(5)
        medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        pcls   = ["podium-1","podium-2","podium-3","",""]
        for i, p in enumerate(podium):
            _, tc, ti = get_tier(p.rating)
            st.markdown(f"""
            <div class="player-row {pcls[i]}">
                <span style="font-size:1.2rem">{medals[i]}</span>
                <span style="font-weight:600;flex:1;margin-left:10px">{p.name}</span>
                <span style="color:{tc};font-weight:700">{ti} {p.rating:.0f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>#### ⬇ Lowest in Queue <span style='font-size:0.73rem;color:#64748b'>(Min Heap)</span>",
                    unsafe_allow_html=True)
        any_q = False
        for mode_nm in MODES:
            lowest = ms.lowest_rated_in_queue(mode_nm)
            if lowest:
                any_q = True
                _, tc, ti = get_tier(lowest.rating)
                mc_c  = mode_color(mode_nm)
                wait  = ms._min_heaps[mode_nm].wait_time(lowest.name)
                st.markdown(f"""
                <div class="player-row" style="border-color:{mc_c}33">
                    <span class="badge badge-{mode_nm.lower()}">{mode_nm}</span>
                    <span style="font-weight:600;margin-left:10px;flex:1">{lowest.name}</span>
                    <span style="color:{tc};font-weight:700;margin-right:10px">{ti} {lowest.rating:.0f}</span>
                    <span style="font-size:0.76rem;color:#64748b">⏱ {fmt_wait(wait)}</span>
                </div>""", unsafe_allow_html=True)
        if not any_q:
            st.markdown('<div style="color:#475569;font-size:0.83rem;padding:6px 0">No players in queue</div>',
                        unsafe_allow_html=True)

    with right:
        st.markdown("#### 📊 Queue by Mode")
        for mode_nm in MODES:
            n   = ms._min_heaps[mode_nm].size
            pct = min(n / max(len(players), 1) * 100, 100)
            mc_c = mode_color(mode_nm)
            st.markdown(f"""
            <div style="margin-bottom:13px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.83rem;font-weight:600">{mode_nm}</span>
                    <span style="font-size:0.8rem;color:#64748b">{n} players</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:6px;height:7px;overflow:hidden">
                    <div style="width:{pct:.1f}%;height:100%;background:{mc_c};border-radius:6px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>#### 📁 Data File")
        if os.path.exists(DATA_FILE):
            sz = os.path.getsize(DATA_FILE)
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                        border-radius:10px;padding:13px">
                <div style="color:#a78bfa;font-weight:600;margin-bottom:3px">📄 {DATA_FILE}</div>
                <div style="color:#64748b;font-size:0.78rem">{sz:,} bytes · JSON</div>
                <div style="color:#64748b;font-size:0.78rem">{len(players)} players · {len(ms._matches)} matches</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No data file yet — register a player to create it.")

    st.markdown("<br>#### ⚔️ Recent Matches", unsafe_allow_html=True)
    recent = list(reversed(done[-5:]))
    if recent:
        for m in recent:
            d1  = (m.rating1_after or m.rating1_before) - m.rating1_before
            d2  = (m.rating2_after or m.rating2_before) - m.rating2_before
            mc  = m.mode.lower()
            c1  = "#22c55e" if d1 >= 0 else "#ef4444"
            c2  = "#22c55e" if d2 >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="match-card" style="display:flex;align-items:center;
                        justify-content:space-between;padding:11px 16px">
                <span style="font-weight:600">{m.player1}
                    <span style="color:{c1};font-size:0.76rem">({d1:+.0f})</span></span>
                <span class="badge badge-{mc}">{m.mode}</span>
                <span class="match-vs">VS</span>
                <span class="badge badge-{mc}">{m.mode}</span>
                <span style="font-weight:600">{m.player2}
                    <span style="color:{c2};font-size:0.76rem">({d2:+.0f})</span></span>
                <span style="color:#64748b;font-size:0.76rem">{fmt_ts(m.timestamp)}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No completed matches yet.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ② — REGISTER PLAYER
# ─────────────────────────────────────────────────────────────────────────────

def page_register(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">➕ Register Player</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Add a new player to the system</div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([2, 1])
    with col_form:
        with st.form("reg_form", clear_on_submit=True):
            name   = st.text_input("Player Name", placeholder="e.g. AlphaWolf")
            rating = st.slider("Starting Rating", 100, 3000, 1200, 25)
            mode   = st.selectbox("Preferred Game Mode", MODES)
            # FIX: removed the unused `_, c, _ = get_tier(rating)` line
            tier_n, tier_c, tier_i = get_tier(rating)
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                        border-radius:8px;padding:10px;margin:6px 0">
                <span style="color:#64748b;font-size:0.78rem">Preview: </span>
                <span style="color:{tier_c};font-weight:700">{tier_i} {tier_n}</span>
                <span style="color:#475569;font-size:0.78rem;margin-left:8px">({rating})</span>
            </div>""", unsafe_allow_html=True)
            if st.form_submit_button("⚡ Register Player", use_container_width=True):
                ok, msg = ms.register_player(name, rating, mode)
                (st.success if ok else st.error)(msg)
                add_activity(msg, "success" if ok else "error")

    with col_info:
        st.markdown("#### Rating Tiers")
        for tname, cfg in TIER_CONFIG.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:9px;margin-bottom:8px;
                        background:rgba(255,255,255,0.02);border-radius:8px;padding:7px 10px">
                <span style="font-size:1.1rem">{cfg['icon']}</span>
                <div>
                    <div style="color:{cfg['color']};font-weight:600;font-size:0.83rem">{tname}</div>
                    <div style="color:#475569;font-size:0.70rem">{cfg['min']}+</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>#### ELO K-Factors", unsafe_allow_html=True)
        for label, val in [("Provisional (<10 games)", K_PROV),
                            ("Established (>=10 games)", K_EST),
                            ("High Rated (>2400)", K_HIGH)]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;
                        font-size:0.78rem;color:#94a3b8">
                <span>{label}</span>
                <span style="color:#a78bfa;font-weight:600">K={val}</span>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Registered Players")
    players = ms.all_players_sorted()
    if players:
        st.dataframe(
            [{"Name": p.name, "Rating": f"{p.rating:.0f}", "Mode": p.mode,
              "W/L/D": f"{p.wins}/{p.losses}/{p.draws}", "Win%": f"{p.win_rate:.1f}%",
              "Registered": fmt_ts(p.registered_at)}
             for p in players],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No players registered yet.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ③ — ALL PLAYERS
# ─────────────────────────────────────────────────────────────────────────────

def page_all_players(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">👥 All Players</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">AVL Tree in-order traversal · O(n) · filter & search</div>',
                unsafe_allow_html=True)
    players = ms.all_players_sorted()
    if not players:
        st.info("No players registered yet.")
        return

    fc1, fc2, fc3 = st.columns([2, 2, 3])
    mf   = fc1.selectbox("Mode",  ["All"] + MODES)
    tf   = fc2.selectbox("Tier",  ["All"] + list(TIER_CONFIG.keys()))
    srch = fc3.text_input("Search by name", placeholder="Type to filter…")

    tier_names = list(TIER_CONFIG.keys())
    filtered   = players
    if mf != "All":
        filtered = [p for p in filtered if p.mode == mf]
    if tf != "All":
        lo  = TIER_CONFIG[tf]["min"]
        idx = tier_names.index(tf)
        hi  = TIER_CONFIG[tier_names[idx - 1]]["min"] - 1 if idx > 0 else 9999
        filtered = [p for p in filtered if lo <= p.rating <= hi]
    if srch:
        filtered = [p for p in filtered if srch.lower() in p.name.lower()]

    st.markdown(f"<div style='color:#64748b;font-size:0.78rem;margin-bottom:9px'>"
                f"Showing {len(filtered)} of {len(players)} players</div>",
                unsafe_allow_html=True)

    for p in filtered:
        tn, tc, ti = get_tier(p.rating)
        mc         = p.mode.lower()
        q_badge    = '<span class="q-pill">IN QUEUE</span>' if p.in_queue else ""
        bar        = min(p.win_rate, 100)
        st.markdown(f"""
        <div class="player-row">
            <div style="flex:2;min-width:100px">
                <div style="font-weight:600">{p.name} {q_badge}</div>
                <div style="font-size:0.70rem;color:#475569">Joined {fmt_ts(p.registered_at)}</div>
            </div>
            <div style="flex:1;text-align:center">
                <span class="badge badge-{mc}">{p.mode}</span>
            </div>
            <div style="flex:1;text-align:center">
                <div style="color:{tc};font-weight:700">{ti} {p.rating:.0f}</div>
                <div style="font-size:0.68rem;color:#475569">{tn}</div>
            </div>
            <div style="flex:2;padding:0 10px">
                <div style="display:flex;justify-content:space-between;font-size:0.70rem;color:#64748b;margin-bottom:2px">
                    <span>W{p.wins}/L{p.losses}/D{p.draws}</span><span>{p.win_rate:.1f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;overflow:hidden">
                    <div style="width:{bar:.1f}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:4px"></div>
                </div>
            </div>
            <div style="flex:1;text-align:right;font-size:0.78rem;color:#64748b">{p.games_played} games</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🔍 AVL Range Query  `O(log n + k)`")
    rc1, rc2, rc3, rc4 = st.columns([2, 2, 2, 1])
    lo_r = rc1.number_input("Min Rating", 100, 3000, 1200, key="rq_lo")
    hi_r = rc2.number_input("Max Rating", 100, 3000, 1800, key="rq_hi")
    rm   = rc3.selectbox("Mode Filter", ["All"] + MODES, key="rq_mode")
    if rc4.button("Search", use_container_width=True):
        res = ms._avl.range_query(lo_r, hi_r, None if rm == "All" else rm)
        if res:
            st.success(f"Found **{len(res)}** players in [{lo_r:.0f}, {hi_r:.0f}]:")
            for p in sorted(res, key=lambda x: -x.rating):
                _, tc, ti = get_tier(p.rating)
                st.markdown(f"- **{p.name}** — <span style='color:{tc}'>{ti} {p.rating:.0f}</span> ({p.mode})",
                            unsafe_allow_html=True)
        else:
            st.warning("No players found in that rating range.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ④ — QUEUE
# ─────────────────────────────────────────────────────────────────────────────

def page_queue(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">⏳ Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">AVL range queries · Min Heap for lowest-rating dispatch</div>',
                unsafe_allow_html=True)

    ctrl, stat_col = st.columns([1, 2])
    with ctrl:
        st.markdown("#### Manage Player")
        names = ms.player_names()
        if not names:
            st.info("Register players first.")
        else:
            sel = st.selectbox("Select Player", [""] + names)
            if sel:
                p = ms.get_player(sel)
                if p:
                    _, tc, ti = get_tier(p.rating)
                    wait = ms._min_heaps[p.mode].wait_time(p.name) if p.in_queue else 0
                    q_str = ("🟢 In Queue" + (f" ({fmt_wait(wait)})" if wait else "")) if p.in_queue else "⚫ Not in Queue"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                                border-radius:10px;padding:13px;margin:7px 0">
                        <div style="font-weight:700;font-size:1rem">{p.name}</div>
                        <div style="color:{tc};margin:3px 0">{ti} {p.rating:.0f}</div>
                        <div style="font-size:0.76rem;color:#64748b">{p.mode} · {q_str}</div>
                    </div>""", unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            if b1.button("➕ Join", use_container_width=True, disabled=not sel):
                ok, msg = ms.join_queue(sel)
                (st.success if ok else st.warning)(msg)
                add_activity(msg, "success" if ok else "warn")
                st.rerun()
            if b2.button("➖ Leave", use_container_width=True, disabled=not sel):
                ok, msg = ms.leave_queue(sel)
                (st.success if ok else st.warning)(msg)
                add_activity(msg, "success" if ok else "warn")
                st.rerun()
            if st.button("⚔️ Auto-Find Match", use_container_width=True, disabled=not sel):
                if not sel:
                    st.warning("Please select a player first.")
                else:
                    ok, msg = ms.find_match(sel)
                    (st.success if ok else st.warning)(msg)
                    add_activity(msg, "match" if ok else "warn")
                    st.rerun()

            if sel:
                p = ms.get_player(sel)
                if p:
                    lowest = ms.lowest_rated_in_queue(p.mode)
                    if lowest:
                        _, tc2, ti2 = get_tier(lowest.rating)
                        wait2 = ms._min_heaps[p.mode].wait_time(lowest.name)
                        st.markdown(f"""
                        <div style="margin-top:10px;background:rgba(56,189,248,0.06);
                                    border:1px solid rgba(56,189,248,0.2);border-radius:8px;padding:9px">
                            <div style="font-size:0.68rem;color:#38bdf8;font-weight:600;
                                        text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px">
                                ⬇ Min Heap · Lowest in {p.mode}
                            </div>
                            <div style="font-weight:600;font-size:0.9rem">{lowest.name}</div>
                            <div style="color:{tc2};font-size:0.83rem">{ti2} {lowest.rating:.0f}</div>
                            <div style="color:#64748b;font-size:0.70rem">Waiting {fmt_wait(wait2)}</div>
                        </div>""", unsafe_allow_html=True)

    with stat_col:
        st.markdown("#### Queue Status")
        for mode_nm in MODES:
            entries = ms.queue_with_wait(mode_nm)
            mc_c    = mode_color(mode_nm)
            lowest  = ms.lowest_rated_in_queue(mode_nm)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:9px;margin-bottom:7px">
                <div style="width:8px;height:8px;border-radius:50%;background:{mc_c}"></div>
                <span style="font-weight:600">{mode_nm}</span>
                <span style="color:#64748b;font-size:0.78rem">({len(entries)} waiting)</span>
            </div>""", unsafe_allow_html=True)

            if entries:
                for p, wait in reversed(entries):
                    _, tc, ti = get_tier(p.rating)
                    is_min    = lowest and p.name == lowest.name
                    min_badge = '<span class="min-pill">MIN</span>' if is_min else ""
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                                border-radius:8px;padding:8px 12px;margin-bottom:5px;
                                display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:500">{p.name} {min_badge}</span>
                        <span style="color:{tc};font-size:0.86rem">{ti} {p.rating:.0f}</span>
                        <span style="color:#64748b;font-size:0.70rem">⏱ {fmt_wait(wait)}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.01);border:1px dashed rgba(255,255,255,0.07);
                            border-radius:8px;padding:10px;text-align:center;
                            color:#475569;font-size:0.78rem;margin-bottom:12px">
                    No players in {mode_nm} queue
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑤ — CREATE MATCH
# ─────────────────────────────────────────────────────────────────────────────

def page_create_match(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">⚔️ Create Match</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Auto-match from queue - players are auto-queued on registration</div>',
                unsafe_allow_html=True)

    # Get all players in queue
    queued = [p for p in ms.all_players_sorted() if p.in_queue]
    
    if len(queued) < 2:
        st.warning("⚠️ Need at least 2 players in queue to create a match.")
        st.info("Players are automatically added to queue when registered.")
        return

    # Show available players in queue
    st.markdown("#### Players in Queue")
    cols = st.columns(min(len(queued), 4))
    for i, p in enumerate(queued):
        _, tc, ti = get_tier(p.rating)
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);
                        border-radius:10px;padding:12px;text-align:center">
                <div style="font-weight:600">{p.name}</div>
                <div style="color:{tc};font-size:0.85rem">{ti} {p.rating:.0f}</div>
                <div style="font-size:0.72rem;color:#64748b">{p.mode}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>")
    
    # Auto-match button - finds best match automatically
    if st.button("⚔️ Auto-Create Match", use_container_width=True, key="auto_match"):
        # Get a player from queue to start matching
        if queued:
            ok, msg = ms.create_match_manual(queued[0].name)
            (st.success if ok else st.error)(msg)
            add_activity(msg, "match" if ok else "error")
            if ok:
                st.balloons()
            st.rerun()

    st.markdown("<br>")
    st.markdown("#### Manual Pairing (Optional)")
    st.markdown("Choose specific players to match against each other.")
    
    names = ms.player_names()
    mc1, mc2 = st.columns(2)
    p1s = mc1.selectbox("Player 1", [""] + names, key="mp1")
    p2s = mc2.selectbox("Player 2", [""] + names, key="mp2")
    
    if p1s and p2s and p1s != p2s:
        p1 = ms.get_player(p1s)
        p2 = ms.get_player(p2s)
        if not p1 or not p2:
            st.error("One or both players not found.")
        else:
            _, tc1, ti1 = get_tier(p1.rating)
            _, tc2, ti2 = get_tier(p2.rating)
            diff = abs(p1.rating - p2.rating)
            fair = diff <= THRESHOLD
            st.markdown(f"""
            <div class="match-card">
                <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center">
                    <div style="text-align:center">
                        <div style="font-size:1.2rem;font-weight:700">{p1.name}</div>
                        <div style="color:{tc1};font-size:0.97rem">{ti1} {p1.rating:.0f}</div>
                        <div style="font-size:0.73rem;color:#64748b">{p1.mode}</div>
                    </div>
                    <div class="match-vs">VS</div>
                    <div style="text-align:center">
                        <div style="font-size:1.2rem;font-weight:700">{p2.name}</div>
                        <div style="color:{tc2};font-size:0.97rem">{ti2} {p2.rating:.0f}</div>
                        <div style="font-size:0.73rem;color:#64748b">{p2.mode}</div>
                    </div>
                </div>
                <div style="text-align:center;margin-top:11px;font-size:0.78rem;
                            color:{'#22c55e' if fair else '#f59e0b'}">
                    {'✅ Fair match' if fair else f'⚠️ Rating gap: {diff:.0f} (threshold ±{THRESHOLD})'}
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button("⚔️ Create Match", use_container_width=True, key="do_manual"):
                ok, msg = ms.create_match_manual(p1s, p2s)
                (st.success if ok else st.error)(msg)
                add_activity(msg, "match" if ok else "error")
                if ok:
                    st.balloons()
                st.rerun()
    elif p1s and p2s and p1s == p2s:
        st.warning("Select two different players.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑥ — MATCH HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def page_match_history(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">📋 Match History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">All matches · resolve pending · ELO updates applied</div>',
                unsafe_allow_html=True)

    pending   = ms.pending_matches()
    completed = ms.completed_matches()
    tab_pend, tab_done = st.tabs([f"⏳ Pending ({len(pending)})", f"✅ Completed ({len(completed)})"])

    with tab_pend:
        if not pending:
            st.info("No pending matches. Create matches on the **Create Match** page.")
        for m in reversed(pending):
            p1 = ms.get_player(m.player1)
            p2 = ms.get_player(m.player2)
            if not p1 or not p2: continue
            _, tc1, ti1 = get_tier(m.rating1_before)
            _, tc2, ti2 = get_tier(m.rating2_before)
            with st.expander(f"Match #{m.match_id}  ·  {m.player1} vs {m.player2}  ·  {m.mode}  ·  {fmt_ts(m.timestamp)}"):
                ec1, ec2, ec3 = st.columns([5, 1, 5])
                ec1.markdown(f"""
                <div style="text-align:center">
                    <div style="font-size:1.3rem;font-weight:700">{m.player1}</div>
                    <div style="color:{tc1};font-size:1rem">{ti1} {m.rating1_before:.0f}</div>
                </div>""", unsafe_allow_html=True)
                ec2.markdown('<div class="match-vs" style="margin-top:6px">VS</div>', unsafe_allow_html=True)
                ec3.markdown(f"""
                <div style="text-align:center">
                    <div style="font-size:1.3rem;font-weight:700">{m.player2}</div>
                    <div style="color:{tc2};font-size:1rem">{ti2} {m.rating2_before:.0f}</div>
                </div>""", unsafe_allow_html=True)
                result = st.radio("Select Result",
                    ["player1", "draw", "player2"],
                    format_func=lambda x: {
                        "player1": f"🏆 {m.player1} Wins",
                        "player2": f"🏆 {m.player2} Wins",
                        "draw":    "🤝 Draw"
                    }[x],
                    horizontal=True, key=f"res_{m.match_id}")
                if st.button(f"✅ Resolve Match #{m.match_id}", key=f"rv_{m.match_id}", use_container_width=True):
                    ok, msg = ms.resolve_match(m.match_id, result)
                    (st.success if ok else st.error)(msg)
                    add_activity(msg, "success" if ok else "error")
                    st.rerun()

    with tab_done:
        if not completed:
            st.info("No completed matches yet.")
        else:
            cf1, cf2 = st.columns(2)
            pf = cf1.text_input("Filter by player", placeholder="Search…", key="hist_p")
            mf = cf2.selectbox("Filter by mode", ["All"] + MODES, key="hist_m")
            shown = list(reversed(completed))
            if pf: shown = [m for m in shown if pf.lower() in m.player1.lower() or pf.lower() in m.player2.lower()]
            if mf != "All": shown = [m for m in shown if m.mode == mf]
            st.markdown(f"<div style='color:#64748b;font-size:0.78rem;margin-bottom:8px'>"
                        f"Showing {len(shown)} of {len(completed)} matches</div>",
                        unsafe_allow_html=True)
            rows = []
            for m in shown:
                winner = {"player1": m.player1, "player2": m.player2, "draw": "Draw"}.get(m.result, "?")
                d1 = (m.rating1_after or m.rating1_before) - m.rating1_before
                d2 = (m.rating2_after or m.rating2_before) - m.rating2_before
                rows.append({"#": m.match_id, "Date": fmt_ts(m.timestamp), "Mode": m.mode,
                              "Player 1": m.player1, "Δ1": f"{d1:+.0f}",
                              "Player 2": m.player2, "Δ2": f"{d2:+.0f}", "Winner": winner})
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑦ — LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_leaderboard(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">🏆 Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Max Heap top-K · O(K log n) · Min Heap lowest per mode</div>',
                unsafe_allow_html=True)

    players = ms.all_players_sorted()
    if not players:
        st.info("No players registered yet.")
        return

    lc1, lc2 = st.columns([3, 1])
    with lc1:
        k = st.slider("Show top N", 3, min(25, len(players)), min(10, len(players)))
    with lc2:
        st.markdown("<br><span style='color:#a78bfa;font-size:0.78rem;font-weight:600'>"
                    "⚙ Max Heap O(K log n)</span>", unsafe_allow_html=True)

    top    = ms.leaderboard(k)
    medals = ["🥇","🥈","🥉"] + [f"#{i+1}" for i in range(3, k+10)]
    pcls   = ["podium-1","podium-2","podium-3"] + [""] * 100
    # FIX: safe division — guard against zero rating
    max_r  = max((p.rating for p in top), default=1) or 1

    for i, p in enumerate(top):
        _, tc, ti = get_tier(p.rating)
        mc  = p.mode.lower()
        bar = min(p.win_rate, 100)
        rel = (p.rating / max_r) * 100
        st.markdown(f"""
        <div class="player-row {pcls[i]}" style="padding:14px 17px">
            <div style="font-size:1.35rem;min-width:46px;text-align:center">{medals[i]}</div>
            <div style="flex:3;margin-left:8px">
                <div style="font-weight:700;font-size:0.97rem">{p.name}</div>
                <div style="display:flex;gap:7px;margin-top:3px;align-items:center">
                    <span class="badge badge-{mc}">{p.mode}</span>
                    <span style="font-size:0.72rem;color:#64748b">{p.games_played} games</span>
                </div>
            </div>
            <div style="flex:2;text-align:center">
                <div style="color:{tc};font-weight:700;font-size:1.1rem">{ti} {p.rating:.0f}</div>
                <div style="font-size:0.68rem;color:#475569">{get_tier(p.rating)[0]}</div>
            </div>
            <div style="flex:2;padding:0 10px">
                <div style="display:flex;justify-content:space-between;font-size:0.70rem;color:#64748b;margin-bottom:2px">
                    <span>W{p.wins}/L{p.losses}/D{p.draws}</span>
                    <span style="color:#a78bfa;font-weight:600">{p.win_rate:.1f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;overflow:hidden">
                    <div style="width:{bar:.1f}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:4px"></div>
                </div>
            </div>
            <div style="flex:2;padding:0 7px">
                <div style="font-size:0.66rem;color:#64748b;margin-bottom:2px">Rating bar</div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;overflow:hidden">
                    <div style="width:{rel:.1f}%;height:100%;background:linear-gradient(90deg,{tc},transparent);border-radius:4px"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🏅 Champion by Mode  &  ⬇ Min Heap · Lowest in Queue")
    mc1, mc2, mc3 = st.columns(3)
    for col, mode_nm in zip([mc1, mc2, mc3], MODES):
        mode_ps = [p for p in players if p.mode == mode_nm]
        best    = mode_ps[0] if mode_ps else None
        lowest  = ms.lowest_rated_in_queue(mode_nm)
        mc_c    = mode_color(mode_nm)
        with col:
            if best:
                _, tc, ti = get_tier(best.rating)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid {mc_c}44;
                            border-radius:12px;padding:14px;text-align:center;margin-bottom:8px">
                    <div style="color:{mc_c};font-weight:700;font-size:0.72rem;
                                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px">🏆 {mode_nm}</div>
                    <div style="font-size:1.1rem;font-weight:700">{best.name}</div>
                    <div style="color:{tc};font-size:0.97rem;margin:3px 0">{ti} {best.rating:.0f}</div>
                    <div style="color:#64748b;font-size:0.72rem">W{best.wins}/L{best.losses}/D{best.draws}</div>
                </div>""", unsafe_allow_html=True)
            if lowest:
                _, tc2, ti2 = get_tier(lowest.rating)
                wait = ms._min_heaps[mode_nm].wait_time(lowest.name)
                st.markdown(f"""
                <div style="background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.2);
                            border-radius:10px;padding:11px;text-align:center">
                    <div style="color:#38bdf8;font-weight:700;font-size:0.68rem;
                                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px">
                        ⬇ Min Heap · Lowest in Queue
                    </div>
                    <div style="font-weight:600;font-size:0.9rem">{lowest.name}</div>
                    <div style="color:{tc2};font-size:0.85rem">{ti2} {lowest.rating:.0f}</div>
                    <div style="color:#64748b;font-size:0.70rem">Waiting {fmt_wait(wait)}</div>
                </div>""", unsafe_allow_html=True)
            if not best and not lowest:
                st.markdown(f"""
                <div style="border:1px dashed rgba(255,255,255,0.07);border-radius:12px;
                            padding:14px;text-align:center;color:#475569;font-size:0.78rem">
                    No {mode_nm} players
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑧ — PLAYER PROFILES
# ─────────────────────────────────────────────────────────────────────────────

AVATAR_OPTIONS = ["🎮", "🔥", "⚡", "💀", "👑", "🦊", "🐉", "🎯", "🚀", "⭐", "🌟", "💎"]

def page_profiles(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">👤 Player Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Customize avatar, bio, and stats</div>', unsafe_allow_html=True)

    players = ms.player_names()
    if not players:
        st.info("No players registered yet.")
        return

    col_sel, col_view = st.columns([1, 2])
    
    with col_sel:
        st.markdown("#### Select Player")
        selected = st.selectbox("Choose player", [""] + players)
        
        if selected:
            profile = ms.get_profile(selected)
            p = ms.get_player(selected)
            
            with st.form("profile_form"):
                st.markdown("#### Edit Profile")
                avatar = st.selectbox("Avatar", AVATAR_OPTIONS, 
                                     index=AVATAR_OPTIONS.index(profile.avatar) if profile and profile.avatar in AVATAR_OPTIONS else 0)
                bio = st.text_area("Bio", value=profile.bio if profile else "", max_chars=200)
                fav_mode = st.selectbox("Favorite Mode", MODES, 
                                        index=MODES.index(profile.favorite_mode) if profile and profile.favorite_mode in MODES else 0)
                country = st.text_input("Country", value=profile.country if profile else "")
                discord = st.text_input("Discord", value=profile.discord if profile else "")
                
                if st.form_submit_button("💾 Save Profile", use_container_width=True):
                    if profile:
                        ok, msg = ms.update_profile(selected, avatar=avatar, bio=bio, 
                                                    favorite_mode=fav_mode, country=country, discord=discord)
                    else:
                        ok, msg = ms.create_profile(selected, avatar=avatar, bio=bio,
                                                    favorite_mode=fav_mode, country=country, discord=discord)
                    (st.success if ok else st.error)(msg)
                    st.rerun()
            
            if st.button("➕ Create Profile", use_container_width=True, disabled=bool(profile)):
                if not profile:
                    ok, msg = ms.create_profile(selected)
                    (st.success if ok else st.error)(msg)
                    st.rerun()

    with col_view:
        st.markdown("#### Profile Preview")
        if selected:
            profile = ms.get_profile(selected)
            p = ms.get_player(selected)
            
            if profile:
                _, tc, ti = get_tier(p.rating)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.2);
                            border-radius:16px;padding:24px;text-align:center">
                    <div style="font-size:4rem;margin-bottom:10px">{profile.avatar}</div>
                    <div style="font-size:1.5rem;font-weight:700">{selected}</div>
                    <div style="color:{tc};font-size:1.1rem;margin:5px 0">{ti} {p.rating:.0f}</div>
                    <div style="color:#64748b;font-size:0.9rem;margin-bottom:10px">{p.mode}</div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;margin:10px 0">
                        <div style="color:#a78bfa;font-weight:600">Bio</div>
                        <div style="color:#c4c9e0;font-size:0.9rem">{profile.bio or 'No bio set'}</div>
                    </div>
                    <div style="display:flex;justify-content:center;gap:20px;margin-top:10px">
                        <div><span style="color:#64748b;font-size:0.78rem">Country</span><br><span>{profile.country or '—'}</span></div>
                        <div><span style="color:#64748b;font-size:0.78rem">Favorite</span><br><span>{profile.favorite_mode}</span></div>
                        <div><span style="color:#64748b;font-size:0.78rem">Discord</span><br><span>{profile.discord or '—'}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No profile yet. Create one to customize!")
        else:
            st.info("Select a player to view/edit their profile.")

    st.divider()
    st.markdown("#### All Players with Profiles")
    profiles_exist = [(name, ms.get_profile(name)) for name in players if ms.get_profile(name)]
    if profiles_exist:
        for name, prof in profiles_exist:
            p = ms.get_player(name)
            _, tc, ti = get_tier(p.rating)
            st.markdown(f"""
            <div class="player-row">
                <span style="font-size:1.5rem">{prof.avatar}</span>
                <div style="flex:1;margin-left:10px">
                    <div style="font-weight:600">{name}</div>
                    <div style="font-size:0.78rem;color:#64748b">{prof.bio[:50]}{'...' if len(prof.bio) > 50 else ''}</div>
                </div>
                <div style="color:{tc}">{ti} {p.rating:.0f}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No profiles created yet.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑨ — TEAMS
# ─────────────────────────────────────────────────────────────────────────────

def page_teams(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">👥 Teams</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Create teams, invite players, team matchmaking</div>', unsafe_allow_html=True)

    players = ms.player_names()
    if len(players) < 2:
        st.warning("Need at least 2 players to create teams.")
        return

    col_create, col_list = st.columns([1, 2])
    
    with col_create:
        st.markdown("#### Create Team")
        with st.form("team_form"):
            team_name = st.text_input("Team Name", placeholder="e.g. Alpha Warriors")
            team_tag = st.text_input("Team Tag", placeholder="e.g. ALP", max_chars=5)
            captain = st.selectbox("Captain", players)
            if st.form_submit_button("🏠 Create Team", use_container_width=True):
                ok, msg = ms.create_team(team_name, captain, team_tag)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()

    with col_list:
        st.markdown("#### All Teams")
        teams = ms.team_names()
        if not teams:
            st.info("No teams created yet.")
        else:
            for tname in teams:
                team = ms.get_team(tname)
                mc = mode_color(team.captain if team else "Ranked")
                st.markdown(f"""
                <div class="player-row" style="border-color:{mc}44">
                    <div style="flex:1">
                        <div style="font-weight:700">{team.name} <span style="color:#64748b;font-size:0.78rem">[{team.tag}]</span></div>
                        <div style="font-size:0.78rem;color:#64748b">Captain: {team.captain} · {len(team.members)} members</div>
                    </div>
                    <div style="text-align:right">
                        <div style="color:#22c55e;font-weight:600">W{team.wins} · L{team.losses}</div>
                        <div style="color:#a78bfa;font-size:0.78rem">{team.win_rate:.1f}%</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                # Show team members
                with st.expander(f"View {team.name} Members"):
                    for member in team.members:
                        p = ms.get_player(member)
                        if p:
                            _, tc, ti = get_tier(p.rating)
                            is_cap = "👑" if member == team.captain else "  "
                            st.markdown(f"- {is_cap} **{member}** — {ti} {p.rating:.0f}")
                    
                    # Add/remove members
                    st.markdown("#### Manage Members")
                    non_members = [pl for pl in players if pl not in team.members]
                    if non_members:
                        add_player = st.selectbox("Add Player", [""] + non_members, key=f"add_{tname}")
                        if st.button("➕ Add", key=f"btn_add_{tname}", disabled=not add_player):
                            ok, msg = ms.join_team(tname, add_player)
                            (st.success if ok else st.error)(msg)
                            st.rerun()
                    
                    # Leave team option
                    leave_player = st.selectbox("Remove Player", [""] + [m for m in team.members if m != team.captain], key=f"leave_{tname}")
                    if st.button("➖ Remove", key=f"btn_leave_{tname}", disabled=not leave_player):
                        ok, msg = ms.leave_team(tname, leave_player)
                        (st.success if ok else st.error)(msg)
                        st.rerun()

    st.divider()
    st.markdown("#### Team Statistics")
    if teams:
        team_stats = []
        for tname in teams:
            team = ms.get_team(tname)
            avg_rating = sum(ms.get_player(m).rating for m in team.members if ms.get_player(m)) / max(len(team.members), 1)
            team_stats.append({"Team": tname, "Members": len(team.members), "W/L": f"{team.wins}/{team.losses}", "Win%": f"{team.win_rate:.1f}", "Avg Rating": f"{avg_rating:.0f}"})
        st.dataframe(team_stats, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑩ — TOURNAMENTS
# ─────────────────────────────────────────────────────────────────────────────

def page_tournaments(ms: MatchmakingSystem):
    st.markdown('<div class="page-title">🏆 Tournaments</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Create brackets, single/double elimination</div>', unsafe_allow_html=True)

    teams = ms.team_names()
    if len(teams) < 2:
        st.warning("Need at least 2 teams to create tournaments.")
        st.info("Create teams first on the Teams page.")
        return

    tab_create, tab_active, tab_completed = st.tabs(["➕ Create", "⏳ Active", "✅ Completed"])

    with tab_create:
        st.markdown("#### Create Tournament")
        with st.form("tournament_form"):
            name = st.text_input("Tournament Name", placeholder="e.g. Summer Championship")
            mode = st.selectbox("Game Mode", MODES)
            fmt = st.selectbox("Format", ["single", "double"], format_func=lambda x: x.title() + " Elimination")
            max_t = st.slider("Max Teams", 4, 16, 8)
            prize = st.text_input("Prize", placeholder="e.g. $100, Trophy, etc.")
            created_by = st.selectbox("Created By", ms.player_names())
            
            if st.form_submit_button("🏆 Create Tournament", use_container_width=True):
                ok, msg = ms.create_tournament(name, mode, fmt, max_t, prize, created_by)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()

    with tab_active:
        active = ms.list_tournaments("open") + ms.list_tournaments("in_progress")
        if not active:
            st.info("No active tournaments.")
        else:
            for t in active:
                mc = mode_color(t.mode)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid {mc}44;
                            border-radius:12px;padding:16px;margin-bottom:12px">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div style="font-size:1.2rem;font-weight:700">{t.name}</div>
                            <div style="color:#64748b;font-size:0.8rem">{t.mode} · {t.format.title()} Elimination</div>
                        </div>
                        <div style="text-align:right">
                            <span class="badge badge-{t.mode.lower()}">{t.mode}</span>
                            <div style="color:#64748b;font-size:0.78rem">{len(t.teams)}/{t.max_teams} teams</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                if t.status == "open":
                    st.markdown("**Registered Teams:**")
                    for team in t.teams:
                        st.markdown(f"- {team}")
                    
                    # Join tournament
                    join_team = st.selectbox("Join with Team", [""] + [tn for tn in teams if tn not in t.teams], key=f"join_{t.tournament_id}")
                    if st.button("➕ Join Tournament", key=f"btn_join_{t.tournament_id}", disabled=not join_team):
                        ok, msg = ms.join_tournament(t.tournament_id, join_team)
                        (st.success if ok else st.error)(msg)
                        st.rerun()
                    
                    if st.button("▶️ Start Tournament", key=f"btn_start_{t.tournament_id}", disabled=len(t.teams) < 2):
                        ok, msg = ms.start_tournament(t.tournament_id)
                        (st.success if ok else st.error)(msg)
                        st.rerun()
                
                elif t.status == "in_progress":
                    st.markdown("#### Bracket")
                    if t.matches:
                        for round_num in sorted(set(m["round"] for m in t.matches)):
                            st.markdown(f"**Round {round_num}**")
                            round_matches = [m for m in t.matches if m["round"] == round_num]
                            for m in round_matches:
                                if m["team2"] == "BYE":
                                    st.markdown(f"- {m['team1']} gets BYE")
                                else:
                                    winner = m["winner"] if m["winner"] else "?"
                                    st.markdown(f"- **{m['team1']}** vs **{m['team2']}** → Winner: {winner}")
                                    
                                    # Resolve match
                                    if not m["winner"]:
                                        winner_sel = st.selectbox(f"Winner: {m['team1']} vs {m['team2']}", 
                                                                 [m["team1"], m["team2"]], 
                                                                 key=f"tw_{t.tournament_id}_{m['team1']}_{m['team2']}")
                                        if st.button("✅ Declare Winner", key=f"dw_{t.tournament_id}_{m['team1']}_{m['team2']}"):
                                            ok, msg = ms.resolve_tournament_match(t.tournament_id, round_num, 
                                                                                   m["team1"], m["team2"], winner_sel)
                                            (st.success if ok else st.error)(msg)
                                            st.rerun()

    with tab_completed:
        completed = ms.list_tournaments("completed")
        if not completed:
            st.info("No completed tournaments.")
        else:
            for t in completed:
                winner = "TBD"
                if t.matches:
                    final = [m for m in t.matches if m["round"] == max(m["round"] for m in t.matches)]
                    if final and final[0]["winner"]:
                        winner = final[0]["winner"]
                
                mc = mode_color(t.mode)
                st.markdown(f"""
                <div class="player-row" style="border-color:{mc}44">
                    <div style="flex:1">
                        <div style="font-weight:700">{t.name}</div>
                        <div style="color:#64748b;font-size:0.78rem">{t.mode} · {t.format.title()}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="color:#ffd700;font-weight:700">🏆 {winner}</div>
                        <div style="color:#64748b;font-size:0.78rem">{len(t.teams)} teams</div>
                    </div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="⚔️ Matchmaking System",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    ms = get_sys()

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">⚔ MATCHMAKER</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">AVL · Max Heap · Min Heap · JSON</div>', unsafe_allow_html=True)

        page = st.radio("nav", options=[
            "🏠  Dashboard",
            "➕  Register Player",
            "👥  All Players",
            "👤  Player Profiles",
            "⏳  Queue",
            "⚔️  Create Match",
            "📋  Match History",
            "🏆  Leaderboard",
            "👥  Teams",
            "🏆  Tournaments",
        ], label_visibility="collapsed")

        st.divider()

        # ── Live stats ─────────────────────────────────────────────────────────
        q_total = sum(ms._min_heaps[mode_nm].size for mode_nm in MODES)
        stat_rows = [
            ("Players",   len(ms._players),           "#a78bfa"),
            ("In Queue",  q_total,                     "#a78bfa"),
            ("Pending",   len(ms.pending_matches()),   "#fbbf24"),
            ("Completed", len(ms.completed_matches()), "#22c55e"),
            ("AVL nodes", ms._avl.size,                "#475569"),
        ]
        st.markdown('<div style="font-size:0.70rem;color:#475569;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:6px">System</div>', unsafe_allow_html=True)
        for lbl, val, vc in stat_rows:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="font-size:0.78rem;color:#64748b">{lbl}</span>
                <span style="font-size:0.78rem;color:{vc};font-weight:600">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Activity log ───────────────────────────────────────────────────────
        st.markdown('<div style="font-size:0.70rem;color:#475569;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:6px">Recent Activity</div>', unsafe_allow_html=True)
        icons = {"success":"✅","error":"❌","info":"ℹ️","warn":"⚠️","match":"⚔️"}
        for act in st.session_state.get("activity", [])[:7]:
            ico = icons.get(act["kind"], "·")
            txt = act["msg"][:44] + ("…" if len(act["msg"]) > 44 else "")
            st.markdown(f'<div style="font-size:0.74rem;color:#64748b;padding:3px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,0.04)">{ico} {txt}</div>',
                        unsafe_allow_html=True)

        st.divider()

        # ── Quick actions ──────────────────────────────────────────────────────
        qa1, qa2 = st.columns(2)
        if qa1.button("🎲 Random", use_container_width=True, help="Random auto-match"):
            names = ms.player_names()
            if len(names) >= 2:
                chosen = random.sample(names, 2)
                for n in chosen:
                    p = ms.get_player(n)
                    if p and not p.in_queue:
                        ms.join_queue(n)
                ok, msg = ms.find_match(chosen[0])
                add_activity(msg, "match" if ok else "warn")
                st.rerun()

        if qa2.button("🔄 Reset", use_container_width=True, help="Wipe all data"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            for key in ["sys", "activity"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── FIX: robust page routing via regex key extraction ─────────────────────
    key = _page_key(page)
    if   key == "Dashboard":       page_dashboard(ms)
    elif key == "Register Player": page_register(ms)
    elif key == "All Players":     page_all_players(ms)
    elif key == "Player Profiles": page_profiles(ms)
    elif key == "Queue":           page_queue(ms)
    elif key == "Create Match":    page_create_match(ms)
    elif key == "Match History":   page_match_history(ms)
    elif key == "Leaderboard":     page_leaderboard(ms)
    elif key == "Teams":           page_teams(ms)
    elif key == "Tournaments":     page_tournaments(ms)


if __name__ == "__main__":
    main()