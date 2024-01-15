import chess


class ItGame:
    def __init__(self, game):
        self._board = game.board().copy()
        self._headers = game.headers.copy()
        self._umake = []
        tmp_lst = []
        for move in game.mainline_moves():
            tmp_lst.append(move)
        self._umake = []
        while tmp_lst:
            self._umake.append(tmp_lst.pop())

    def unmake_move(self):
        if self._board.move_stack:
            self._umake.append(self._board.pop())
            return True
        return False

    def make_move(self):
        if self._umake:
            assert self._umake[-1] in self._board.legal_moves
            # print("making move")
            self._board.push(self._umake.pop())

    def in_check(self):
        return self._board.is_check()

    def __last_is_capture(self):
        if self._board.move_stack and self._board.halfmove_clock == 0:
            self.unmake_move()
            res = self.__next_is_capture()
            self.make_move()
            return res
        return False

    def __next_is_check(self):
        if self._umake:
            return self._board.gives_check(self._umake[-1])
        return False

    def __next_is_capture(self):
        if self._umake:
            return self._board.is_capture(self._umake[-1])
        return False

    def __next_is_en_passant(self):
        if self._umake:
            return self._board.is_en_passant(self._umake[-1])
        return False

    def to_start(self):
        assert (self._umake != None)
        while (self.unmake_move()):
            continue

    def castling_rights(self):
        return self._board.castling_rights

    def halfmove_clock(self):
        return self._board.halfmove_clock

    def is_quiet(self):
        return not self.in_check() and not self.__next_is_en_passant() and (
                    (self._board.halfmove_clock or not self.__last_is_capture()) or not (
                        self.__next_is_check() or self.__next_is_capture()))

    def is_final_position(self):
        return len(self._umake) == 0

    def get_board_copy(self):
        return self._board.copy()

    def headers(self):
        return self._headers

    def fen(self):
        return self._board.fen()

    def print_board(self):
        print(self._board)

