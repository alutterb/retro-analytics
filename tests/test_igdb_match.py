import pytest

from retro_analytics import igdb_match


def game(id, name, game_type=0, votes=0, alternatives=()):
    return {'id': id, 'name': name, 'game_type': game_type, 'total_rating_count': votes,
            'alternative_names': [{'name': a} for a in alternatives]}


def best(title, games, console='nes'):
    found = igdb_match.match(title, igdb_match.candidates(games), console)
    return found and found[1]['id']


def test_norm_folds_case_accents_punctuation_and_numerals():
    assert igdb_match.norm("Pokémon: Brothers & Sisters II") == "pokemon bros and sisters 2"
    assert igdb_match.norm("The Legend of Zelda") == "legend of zelda"
    assert igdb_match.norm("Mega Man X") == "mega man x"


@pytest.mark.parametrize('title, name', [
    ("Super Mario Bros 3", "Super Mario Bros. 3"),
    ("Zelda Phantom Hourglass", "The Legend of Zelda: Phantom Hourglass"),
    ("Luigi's Mansion [Player's Choice]", "Luigi's Mansion"),
    ("Spiderman", "Spider-Man"),
    ("Phantasy Star IV", "Phantasy Star IV: The End of the Millennium"),
    ("Kirby Tilt and Tumble", "Kirby Tilt 'n' Tumble"),
])
def test_matches_title_variants(title, name):
    assert best(title, [game(1, name), game(2, "Unrelated Game")]) == 1


@pytest.mark.parametrize('title, name', [
    ("Mega Man X", "Mega Man 10"),
    ("Silent Hill 2", "Silent Hill 4"),
    ("Dragon's Lair II", "Dragon's Lair"),
])
def test_numbers_must_agree(title, name):
    assert best(title, [game(1, name)]) is None


def test_console_number_is_not_a_sequel_number():
    assert best("Kirby 64 Crystal Shards", [game(1, "Kirby: The Crystal Shards")], console="nintendo-64") == 1
    assert best("Kirby 64 Crystal Shards", [game(1, "Kirby: The Crystal Shards")], console="nes") is None


def test_alternative_names_match():
    assert best("Dino Force", [game(1, "Dinoforce Japan", alternatives=["Dino Force"])]) == 1


def test_mods_are_never_chosen():
    assert best("Majoras Mask", [game(1, "Majora's Mask", game_type=5)]) is None


def test_main_game_wins_a_tie_with_its_port():
    games = [game(2, "Tetris Worlds", game_type=11, votes=500), game(1, "Tetris Worlds", game_type=0, votes=3)]
    assert best("Tetris Worlds", games) == 1


def test_weak_match_is_rejected():
    assert best("Golden Axe Warrior", [game(1, "Golden Sun")]) is None
