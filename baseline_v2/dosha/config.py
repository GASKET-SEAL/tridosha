"""All Ayurvedic constants of the 5 Nov plan in one place. Source notes mark what is verified vs draft."""
import numpy as np
DOSHAS = ("vata", "pitta", "kapha")
# Five guna axes (heat, oily, heavy, mobile, sharp) -> rows Vata, Pitta, Kapha.
# Qualities per Charaka / Ashtanga Hridaya (verified against review articles). Numeric values are DRAFT (panel sign-off needed).
AXES = ("heat", "oily", "heavy", "mobile", "sharp")
G = np.array([[-.6, -1, -.8, 1, .1], [1, .3, -.2, .3, 1], [-.5, 1, 1, -1, -.8]])
G_NORM = 2.2
RHO = np.array([.55, .65, .80])            # persistence per dosha (Vata fast, Kapha slow) - DRAFT
TANH_C = 1.5                               # bound on log-excess
BETA = 1.0                                 # susceptibility "like increases like" - DRAFT
ENV_SCALE, RITU_SCALE = 0.35, 0.15
AGNI_GAIN, AGNI_SMOOTH = 0.4, 0.5
# Ritu: start day-of-year (Shishira, Vasanta, Grishma, Varsha, Sharad, Hemanta). Vata/Pitta rows agree across sources;
# Kapha accumulation is Shishira in some sources and Hemanta in others, so both carry it (panel to decide).
RB = [15, 74, 135, 196, 258, 319]
RT = np.array([[0, 0, .5], [0, 0, 1], [.5, 0, -.5], [1, .5, 0], [-.5, 1, 0], [0, -.5, .5]])
RITU_NAMES = ("Shishira", "Vasanta", "Grishma", "Varsha", "Sharad", "Hemanta")
SLEEP_BAND = (7.0, 9.0)                    # AASM/SRS consensus: 7+ h for adults (Watson 2015)
# Dosha clock windows (6-10 Kapha, 10-2 Pitta, 2-6 Vata) -> energy slot owner index
ENERGY_OWNER = {"em": 2, "ea": 1, "ee": 0}
# Activity MET ranges (light, vigorous) from the 2024 Adult Compendium: yoga hatha 2.3 .. power 4.0, nadisodhana 2.0,
# resistance 3.5..6.0, sports cricket 4.8 .. basketball 8.0, walking 3.8..4.8, HIIT 7.0..11.0
MET = {"gym": (3.5, 6.0), "sports": (4.8, 8.0), "yoga": (2.0, 4.0), "pran": (2.0, 2.0), "walk": (3.8, 4.8), "hiit": (7.0, 11.0)}
ACT_GUNA = {"gym": [.3, -.3, -.6, .3, .3], "sports": [.8, -.4, -.6, .8, .5], "yoga": [0, .3, 0, -.8, -.2],
            "pran": [-.3, .2, -.2, -.6, -.3], "walk": [.1, 0, -.3, .2, 0], "hiit": [1, -.5, -.7, 1, .6]}
MET_SCALE = 250.0
MOOD = {"anxious": [1, 0, 0], "irritable": [0, 1, 0], "withdrawn": [0, 0, 1], "tired": [.3, 0, .7], "calm": [0, 0, 0], "happy": [-.1, -.1, -.1]}
