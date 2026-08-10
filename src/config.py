name = None
log_file = None
device = None
rec = None
tablebase = None
# Probability that a tablebase position which is *not* the entry point into the endgame
# gets its label rewritten by Syzygy. 1.0 reproduces the historical "relabel everything"
# behavior. Lower values keep the game's actual outcome instead, preserving the practical
# difficulty of converting the endgame. See data.data_from_fen_res_set.
tb_relabel_prob = 1.0
# Drop games whose decisive result no chess rule accounts for (time forfeits). Needed for
# the desk_v303..v317 PGNs, where cluster contention produced forfeit rates from 4% to 52%.
# See data.is_abnormal_termination.
drop_abnormal = False
# activation_hook = None
