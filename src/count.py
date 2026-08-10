import collections

tb_changed = [0, 0, 0, 0, 0, 0, 0]
total_tb_queries = [0, 0, 0, 0, 0, 0, 0]
total_changed = 0
total_games = 0
skipped_startpos = 0
skipped_abnormal = 0
repaired_abnormal = 0
# Games dropped by game_filter.check_game, keyed by the failed check.
rejected = collections.Counter()
# Games whose Result disagreed with a terminal final position and was corrected from it.
relabelled_terminal = 0
