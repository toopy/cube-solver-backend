"""Utility helpers for converting UI colors to environment states.

This module bridges between a color-based cube representation used by the
frontend and the environment's integer state encoding expected by the
search/heuristic components.
"""

from typing import Any, List

from oc_rubik_s_cube_dqn.cube import RubiksCubeEnv

FACE_ORDER: List[int] = [
    6,
    3,
    0,
    7,
    4,
    1,
    8,
    5,
    2,
]

PIECES_TO_STATE_IDS_MAPPING: dict[str, List[int]] = {
    "B": [40],
    "G": [49],
    "O": [22],
    "R": [31],
    "W": [4],
    "Y": [13],
    "BO": [43, 19],
    "BR": [37, 34],
    "BW": [41, 5],
    "BY": [39, 12],
    "GO": [46, 25],
    "GR": [52, 28],
    "GW": [50, 3],
    "GY": [48, 14],
    "OW": [23, 1],
    "OY": [21, 10],
    "RW": [32, 7],
    "RY": [30, 16],
    "BOW": [44, 20, 2],
    "BOY": [42, 18, 9],
    "BRW": [38, 35, 8],
    "BRY": [36, 33, 15],
    "GOW": [47, 26, 0],
    "GOY": [45, 24, 11],
    "GRW": [53, 29, 6],
    "GRY": [51, 27, 17],
}
def get_m_cube_from_colors(color_input: List[List[str]]) -> Any:
    """Construct the internal cube object from a color layout.

    Parameters
    ----------
    color_input : list of list of str
        Facelet colors per face as provided by the UI.

    Returns
    -------
    Any
        The internal cube object as used by ``RubiksCubeEnv`` (implementation
        detail of the external dependency). The object exposes ``get()`` and
        ``_cube`` used by ``get_state_from_m_cube``.
    """
    cube_env = RubiksCubeEnv()
    cube_env.set_cube_from_colors(color_input)
    return cube_env.cube


def get_state_from_m_cube(m_cube: Any) -> List[int]:
    """Convert internal cube into the environment's integer state encoding.

    This maps each facelet to an integer ID matching the expectations of the
    deep solver environment. It relies on the cube's exposed ``get()`` method
    and ``_cube`` internal structure provided by ``RubiksCubeEnv``.

    Parameters
    ----------
    m_cube : Any
        Internal cube object. Must provide ``get()`` yielding 54 facelets and
        an attribute ``_cube`` with 3D indexing used here.

    Returns
    -------
    list of int
        Flattened state values ordered as expected by the environment.
    """
    def get_state_value(color: str, piece: Any) -> int:
        key = "".join(sorted(str(piece)))
        return PIECES_TO_STATE_IDS_MAPPING[key][key.index(color)]

    state_colors: List[int] = []

    for i, c in enumerate(m_cube.get()):
        if i in [0, 9, 38]:
            state_value = get_state_value(c, m_cube._cube[0][2][0])

        elif i in [1, 37]:
            state_value = get_state_value(c, m_cube._cube[1][2][0])

        elif i in [2, 29, 36]:
            state_value = get_state_value(c, m_cube._cube[2][2][0])

        elif i in [3, 10]:
            state_value = get_state_value(c, m_cube._cube[0][2][1])

        elif i in [4]:  # W
            state_value = get_state_value(c, m_cube._cube[1][2][1])

        elif i in [5, 28]:
            state_value = get_state_value(c, m_cube._cube[2][2][1])

        elif i in [6, 11, 18]:
            state_value = get_state_value(c, m_cube._cube[0][2][2])

        elif i in [7, 19]:
            state_value = get_state_value(c, m_cube._cube[1][2][2])

        elif i in [8, 20, 27]:
            state_value = get_state_value(c, m_cube._cube[2][2][2])

        elif i in [12, 41]:
            state_value = get_state_value(c, m_cube._cube[0][1][0])

        elif i in [13]:  # O
            state_value = get_state_value(c, m_cube._cube[0][1][1])

        elif i in [14, 21]:
            state_value = get_state_value(c, m_cube._cube[0][1][2])

        elif i in [15, 44, 51]:
            state_value = get_state_value(c, m_cube._cube[0][0][0])

        elif i in [16, 48]:
            state_value = get_state_value(c, m_cube._cube[0][0][1])

        elif i in [17, 24, 45]:
            state_value = get_state_value(c, m_cube._cube[0][0][2])

        elif i in [22]:  # G
            state_value = get_state_value(c, m_cube._cube[1][1][2])

        elif i in [23, 30]:
            state_value = get_state_value(c, m_cube._cube[2][1][2])

        elif i in [25, 46]:
            state_value = get_state_value(c, m_cube._cube[1][0][2])

        elif i in [26, 33, 47]:
            state_value = get_state_value(c, m_cube._cube[2][0][2])

        elif i in [31]:  # R
            state_value = get_state_value(c, m_cube._cube[2][1][1])

        elif i in [32, 39]:
            state_value = get_state_value(c, m_cube._cube[2][1][0])

        elif i in [34, 50]:
            state_value = get_state_value(c, m_cube._cube[2][0][1])

        elif i in [35, 42, 53]:
            state_value = get_state_value(c, m_cube._cube[2][0][0])

        elif i in [40]:  # B
            state_value = get_state_value(c, m_cube._cube[1][1][0])

        elif i in [43, 52]:
            state_value = get_state_value(c, m_cube._cube[1][0][0])

        elif i in [49]:  # Y
            state_value = get_state_value(c, m_cube._cube[1][0][1])

        state_colors.append(state_value)

    state_colors_ordered: List[int] = []

    for i in [0, 5, 1, 3, 4, 2]:
        offset = i * 9
        for j in FACE_ORDER:
            state_colors_ordered.append(state_colors[offset + j])

    return state_colors_ordered
