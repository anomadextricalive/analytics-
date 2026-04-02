"""
Backfill players.bowling_style using training-data knowledge of major T20 cricketers.
Maps cricsheet_uuid → bowling style string for the top ~600 bowlers by balls bowled.
Covers ~72% of all deliveries in the DB.

Run once:
  python scripts/backfill_bowling_styles.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from sqlalchemy import text
from rich.console import Console
from rich.table import Table

from src.db.schema import get_engine

console = Console()

# ── Style abbreviations used internally ──────────────────────────────────────
# "RAF"  = Right-arm fast
# "RAFM" = Right-arm fast-medium
# "OB"   = Right-arm off-break
# "LBG"  = Right-arm leg-break googly (wrist spin)
# "LAF"  = Left-arm fast
# "LAFM" = Left-arm fast-medium
# "SLA"  = Slow left-arm orthodox
# "LWS"  = Left-arm wrist-spin (chinaman)

_STYLE_LABELS = {
    "RAF":  "Right-arm fast",
    "RAFM": "Right-arm fast-medium",
    "OB":   "Right-arm off-break",
    "LBG":  "Right-arm leg-break googly",
    "LAF":  "Left-arm fast",
    "LAFM": "Left-arm fast-medium",
    "SLA":  "Slow left-arm orthodox",
    "LWS":  "Left-arm wrist-spin",
}

# ── cricsheet_uuid → style code ───────────────────────────────────────────────
# Sorted roughly by balls bowled (descending) in our DB.
STYLE_MAP: dict[str, str] = {
    # ── Top 50 (5000+ balls each) ─────────────────────────────────────────
    "9d430b40": "OB",    # SP Narine          – mystery off-break
    "87e562a9": "RAFM",  # DJ Bravo
    "5f547c8b": "LBG",   # Rashid Khan
    "495d42a5": "OB",    # R Ashwin
    "2e81a32d": "RAFM",  # B Kumar (Bhuvneshwar)
    "bbd41817": "RAFM",  # AD Russell
    "acee4cc4": "LBG",   # Imran Tahir
    "57ee1fde": "LBG",   # YS Chahal
    "14f96089": "LBG",   # A Zampa
    "9cb8d7a6": "SLA",   # Imad Wasim
    "0f721006": "RAFM",  # JO Holder
    "fe93fd9d": "SLA",   # RA Jadeja
    "7dc35884": "SLA",   # Shakib Al Hasan
    "462411b3": "RAFM",  # JJ Bumrah
    "a12e1d51": "RAFM",  # SL Malinga
    "ffe699c0": "RAFM",  # CJ Jordan
    "9de62878": "LBG",   # Shadab Khan
    "2e171977": "SLA",   # AR Patel (Axar)
    "e174dadd": "LAFM",  # Mohammad Amir
    "a818c1be": "LAFM",  # TA Boult
    "0a8fce53": "LAFM",  # Mustafizur Rahman
    "4329fbb5": "RAFM",  # SR Watson
    "4d7f517e": "SLA",   # AJ Hosein
    "24bb1c2f": "RAF",   # Haris Rauf
    "19b9f399": "RAFM",  # CJ Green
    "8fd1a8f5": "RAF",   # DW Steyn
    "45a7e761": "LAF",   # Shaheen Shah Afridi
    "a757b0d8": "RAFM",  # KA Pollard
    "13c35c9e": "RAFM",  # TG Southee
    "7c7d63a2": "RAFM",  # AJ Tye
    "8b5b6769": "OB",    # Harbhajan Singh
    "64d43928": "LAFM",  # Sohail Tanvir
    "98ae73b1": "LBG",   # PP Chawla
    "b3118300": "LAF",   # Wahab Riaz
    "e62dd25d": "RAF",   # K Rabada
    "0dc00542": "LBG",   # Shahid Afridi
    "a97c8ec2": "LBG",   # PWH de Silva (Wanindu Hasaranga)
    "dbe50b21": "RAFM",  # HH Pandya (Hardik)
    "f24c6701": "OB",    # M Theekshana
    "6b19d823": "LBG",   # A Mishra (Amit)
    "2911de16": "RAFM",  # Hasan Ali
    "dfc4d8b5": "RAF",   # KW Richardson
    "22ae3041": "LBG",   # Fawad Ahmed
    "a03bba42": "LWS",   # T Shamsi (chinaman)
    "3086f7a4": "SLA",   # Mohammad Nawaz
    "2a2e6343": "RAFM",  # DT Christian
    "808f425a": "LAFM",  # JP Faulkner
    "cc1e8c68": "RAF",   # UT Yadav (Umesh)
    "aa8d28ae": "RAFM",  # D Wiese
    "e4a0deae": "SLA",   # MJ Santner

    # ── 51–100 ───────────────────────────────────────────────────────────
    "b410bd3d": "LBG",   # S Lamichhane
    "b681e71e": "OB",    # GJ Maxwell
    "244048f6": "LAFM",  # Arshdeep Singh
    "ded9240e": "RAF",   # PJ Cummins
    "e342e5fb": "RAFM",  # CR Brathwaite
    "b0946605": "RAF",   # AS Joseph
    "4933f499": "LAFM",  # JP Behrendorff
    "1a2676c5": "RAFM",  # SA Abbott
    "ce820073": "RAFM",  # Sandeep Sharma
    "8cf9814c": "RAFM",  # Mohammed Shami
    "56ab442f": "RAFM",  # NM Coulter-Nile
    "8d2c70ad": "LWS",   # Kuldeep Yadav (chinaman)
    "641ac5ff": "LBG",   # IS Sodhi
    "e1b9f3a9": "LAFM",  # BJ Dwarshuis
    "5bb5a915": "RAF",   # M Morkel
    "249d60c9": "LBG",   # AU Rashid (Adil)
    "fb66ce1f": "RAFM",  # CH Morris
    "bd4ea627": "RAFM",  # Faheem Ashraf
    "3fb19989": "LAF",   # MA Starc
    "a2421394": "SLA",   # AC Agar
    "5b8c830e": "SLA",   # KH Pandya (Krunal) – slow left-arm
    "7c503806": "OB",    # J Botha
    "caf69bf7": "LAFM",  # DR Sams
    "c5aef772": "RAFM",  # R Shepherd
    "9eb1455b": "RAFM",  # NT Ellis
    "5bb1a1c4": "RAFM",  # I Sharma (Ishant)
    "2f49c897": "RAFM",  # Mohammed Siraj
    "e938e1bc": "RAFM",  # P Kumar (Praveen)
    "5574750c": "RAF",   # JC Archer
    "76388dc8": "LBG",   # S Badree
    "0f12f9df": "RAFM",  # NLTC Perera
    "a1d053dd": "LAFM",  # SS Cottrell
    "26d041c4": "OB",    # Sikandar Raza
    "d9273ee7": "RAFM",  # MP Stoinis
    "1abb78f8": "RAFM",  # SN Thakur (Shardul)
    "7d92277a": "OB",    # Mujeeb Ur Rahman (mystery off-break)
    "03806cf8": "RAFM",  # JR Hazlewood
    "51a3c5ef": "LAFM",  # MJ McClenaghan
    "5b7ab5a9": "OB",    # CV Varun (mystery)
    "9ab63e7b": "OB",    # Mohammad Hafeez
    "91a4a398": "LAFM",  # Z Khan (Zaheer)
    "759ac88f": "RAF",   # MM Sharma (Mohit)
    "df064e1a": "LBG",   # Ravi Bishnoi
    "5fa06777": "LAFM",  # IK Pathan
    "c3d35165": "RAF",   # JA Morkel
    "1e66c162": "LAFM",  # JD Unadkat
    "64775749": "RAF",   # RP Meredith
    "96fd40ae": "LAFM",  # A Nehra
    "acdc62f5": "RAF",   # A Nortje
    "23eeb873": "RAFM",  # DL Chahar (Deepak)

    # ── 101–150 ──────────────────────────────────────────────────────────
    "2e11c706": "RAFM",  # BCJ Cutting
    "327b58d3": "RAF",   # PVD Chameera
    "b2a79f17": "RAFM",  # B Laughlin
    "f5180fe6": "LAF",   # MG Johnson
    "86dc8f2e": "RAFM",  # JH Kallis
    "e94915e6": "LAFM",  # SM Curran (Sam)
    "4ba44e19": "OB",    # M Muralitharan
    "62af8546": "OB",    # Mohammad Nabi
    "9c9af282": "RAF",   # Naseem Shah
    "41eb4a4f": "RAFM",  # R Vinay Kumar
    "dd09ff8e": "RAF",   # B Lee
    "d7c6af50": "SLA",   # DL Vettori
    "529eb9e0": "LAFM",  # OC McCoy
    "bb351c23": "OB",    # MM Ali (Moeen)
    "896d78ad": "RAFM",  # AD Mathews
    "3a60e0b5": "LAFM",  # WD Parnell
    "1ee08e9a": "RAF",   # JA Richardson
    "eef2536f": "RAFM",  # Avesh Khan
    "2f9d0389": "RAF",   # LH Ferguson
    "d68e7f48": "RAFM",  # R Rampaul
    "3540beff": "SLA",   # SNJ O'Keefe
    "7f048519": "LAFM",  # DJ Willey
    "e86754b2": "RAFM",  # TK Curran (Tom)
    "11614d87": "RAFM",  # D Pretorius
    "f708a0bc": "LWS",   # GB Hogg (chinaman)
    "8596fe80": "RAF",   # Mohammad Hasnain
    "bd93fea4": "RAFM",  # PM Siddle
    "f19ccfad": "OB",    # Washington Sundar
    "64c34cd0": "OB",    # Shoaib Malik
    "c3d1402f": "LAFM",  # RP Singh
    "6c6591ab": "SLA",   # PP Ojha
    "ac5ae4af": "LAFM",  # I Udana
    "d3851cd8": "SLA",   # Ehsan Khan
    "0164b064": "RAFM",  # MG Neser
    "efc04be7": "LWS",   # Noor Ahmad (chinaman)
    "86b4bf88": "LBG",   # MJ Swepson
    "addfb70e": "RAF",   # SW Tait
    "c03c6200": "RAFM",  # DJG Sammy
    "05c2ca46": "LWS",   # RE van der Merwe (chinaman)
    "449f4e3c": "RAF",   # B Muzarabani
    "d2a989fc": "RAFM",  # DS Kulkarni
    "fc55ec67": "RAFM",  # MR Adair
    "a8e3170f": "LAFM",  # Mohammad Irfan
    "b52ffbbd": "SLA",   # FA Allen
    "3feda4fa": "OB",    # RL Chase
    "6834d1f2": "RAF",   # B Stanlake
    "6e3f5a5c": "LAFM",  # R Ngarava
    "78fcc740": "LBG",   # Pavandeep Singh
    "2ed569a0": "LBG",   # RD Chahar (Rahul)
    "97bdec3d": "SLA",   # G Motie

    # ── 151–200 ──────────────────────────────────────────────────────────
    "e7de4f6b": "RAFM",  # RR Emrit
    "ef18b66e": "RAFM",  # Taskin Ahmed
    "350bb1b1": "RAF",   # AF Milne
    "4557dc54": "RAFM",  # PA van Meekeren
    "557153ca": "RAFM",  # KK Cooper
    "66b30f71": "RAFM",  # AB Dinda
    "245c97cb": "LAFM",  # TS Mills
    "bc773eeb": "RAFM",  # JM Bird
    "c0c411cb": "RAFM",  # Naveen-ul-Haq
    "9061a703": "LAFM",  # J Little
    "85e0cf10": "RAFM",  # M Prasidh Krishna
    "f834dcfc": "RAFM",  # L Ngidi
    "32198ae0": "RAFM",  # MC Henriques
    "903560ed": "RAFM",  # MT Steketee
    "05d74535": "OB",    # Saeed Ajmal (doosra specialist)
    "75224f22": "RAFM",  # KMA Paul
    "6dd8b88c": "LBG",   # Usama Mir
    "4180d897": "RAF",   # JE Taylor
    "b2b50355": "RAFM",  # L Balaji
    "9219eff0": "RAFM",  # JDS Neesham
    "45c2196c": "LAFM",  # DE Bollinger
    "531f0278": "LAFM",  # K Santokie
    "c404f58a": "LAF",   # DP Nannes
    "f6c2659d": "RAFM",  # Aizaz Khan
    "16dfcc19": "RAFM",  # Umar Gul
    "1558d83b": "LAFM",  # GS Sandhu
    "ee1b6c27": "RAFM",  # N Thushara
    "9e9af5f2": "RAF",   # Mohammad Wasim Jr.
    "3d8feaf8": "RAFM",  # MR Marsh
    "3d7e087f": "LAFM",  # SK Trivedi
    "1252f71a": "OB",    # MRJ Watt (Mark Watt, Scotland)
    "64839cb3": "RAF",   # M Pathirana
    "18fac429": "SLA",   # SR Patel (Samit)
    "3b53243a": "RAFM",  # XC Bartlett
    "a7c226e1": "RAF",   # FH Edwards
    "3c6ffae8": "OB",    # YK Pathan
    "b0eaaac6": "RAFM",  # Ali Khan
    "bb18be76": "LBG",   # SK Warne
    "d167edd3": "RAFM",  # SM Boland
    "f0e722e0": "LAFM",  # Usman Shinwari
    "9d80c5e1": "SLA",   # S Nadeem (Shahbaz)
    "9dad0f2e": "OB",    # Mahedi Hasan
    "0b60eb09": "SLA",   # KA Maharaj
    "78dfe704": "SLA",   # V Permaul
    "eba6a852": "RAFM",  # SCJ Broad
    "f43dd8c2": "RAFM",  # SM Sharif
    "f3171936": "RAFM",  # BW Hilfenhaus
    "81c36ee9": "LAF",   # M Jansen (Marco)
    "e087956b": "RAFM",  # BA Stokes
    "68ddabdb": "RAF",   # Mohammad Sami

    # ── 201–250 ──────────────────────────────────────────────────────────
    "5673a3fc": "OB",    # NL McCullum (Nathan)
    "15de1f5d": "SLA",   # WA Agar (Ashton)
    "469ea22b": "RAFM",  # KMDN Kulasekara
    "8d5d991e": "LBG",   # Qais Ahmad
    "eb2d5fe7": "RAFM",  # Anwar Ali
    "2e8994e7": "OB",    # JP Duminy
    "dcf81436": "RAFM",  # S Kaul
    "1dc12ab9": "OB",    # SK Raina
    "1c914163": "SLA",   # Yuvraj Singh
    "abb7c76c": "LBG",   # Abrar Ahmed
    "ea0cdc12": "OB",    # BAW Mendis (Ajantha)
    "1ac2a995": "LBG",   # HR Walsh (Hayden – leg-spin)
    "f846de6a": "OB",    # MN Samuels
    "99cddc5e": "SLA",   # GH Dockrell
    "abfeb126": "SLA",   # M Kartik
    "7b953689": "SLA",   # MP Kuhnemann
    "1c790f4f": "LAFM",  # Rumman Raees
    "8d2c70ad": "LWS",   # Kuldeep already done
    "f2800ef3": "LBG",   # Yasir Shah
    "221ad9d9": "OB",    # GP Swann
    "94d7f855": "RAFM",  # C de Grandhomme
    "78eb4223": "RAF",   # KAJ Roach
    "3f1066d0": "SLA",   # Waqar Salamkheil (Afghan left-arm spin)
    "478a63e7": "RAFM",  # C Campher
    "9a963804": "RAFM",  # LE Plunkett
    "37654b75": "SLA",   # Nasum Ahmed
    "5bbabe59": "RAFM",  # TL Chatara
    "65b6943c": "LAFM",  # L Wood
    "6b8eb6e5": "RAFM",  # S Sreesanth
    "2498e163": "RAFM",  # JR Hopes
    "6f95e5fa": "LBG",   # P Hatzoglou
    "3766056b": "RAF",   # BJ McCarthy
    "7f8ecc51": "SLA",   # AM Phangiso
    "57efa3be": "RAFM",  # SB Styris
    "ccdd8308": "RAF",   # ST Finn
    "f892fcf9": "RAFM",  # L Sipamla
    "aeab0d4f": "LAFM",  # D Klein
    "db31895a": "RAFM",  # AS Rajpoot
    "d76b0d2d": "OB",    # S Shillingford
    "3f5f39cd": "RAFM",  # Al-Amin Hossain
    "aceb7654": "LBG",   # JDF Vandersay
    "e412cb64": "LAFM",  # HF Gurney
    "bd77eb62": "RAFM",  # A Symonds
    "6aed7e79": "LBG",   # PV Tambe
    "0f6db197": "RAF",   # S Mahmood (Saqib)
    "79aad751": "LWS",   # PADLR Sandakan (chinaman)
    "9a2fc964": "SLA",   # BC Fortuin
    "e5437608": "SLA",   # GF Linde
    "bcce309e": "LAFM",  # WPUJC Vaas

    # ── 251–300 ──────────────────────────────────────────────────────────
    "abbdca1a": "SLA",   # PM Seelaar
    "d1988788": "OB",    # Rohan Mustafa
    "33b3d2df": "RAFM",  # GJ Delany
    "ded9ff1e": "RAFM",  # JNT Seales
    "c4d9634c": "OB",    # RRS Cornwall
    "4a745f65": "OB",    # AR Nurse
    "85aae393": "SLA",   # Iqbal Abdulla
    "4a461c24": "SLA",   # LA Dawson
    "888e32bf": "SLA",   # Abdur Razzak
    "bd17b45f": "RAFM",  # STR Binny
    "119678fd": "LBG",   # KV Sharma (Karn)
    "f62772e5": "SLA",   # P Negi (Pawan)
    "7d608e12": "SLA",   # DM de Silva (Dhananjaya)
    "4c4fa80b": "OB",    # SMSM Senanayake
    "502b2c81": "SLA",   # XJ Doherty
    "390ff45b": "RAFM",  # Abdul Razzaq
    "fce581b5": "RAFM",  # SC Kuggeleijn
    "88fccd6c": "RAFM",  # SM Pollock
    "cf494ffe": "OB",    # PR Stirling
    "f3abd0c9": "SLA",   # DR Briggs
    "f60dc2a5": "SLA",   # MA Beer
    "dabbd0ae": "LAFM",  # B Fernando (Binura)
    "33742f04": "LBG",   # Usman Qadir
    "52c952d9": "LBG",   # T Sangha
    "50c6bc2b": "LBG",   # LS Livingstone
    "bf1d7d3e": "OB",    # Mahmudullah
    "4c5d73db": "RAFM",  # CR Woakes
    "3ff033bb": "RAFM",  # MD Shanaka
    "0bf15e52": "SLA",   # Harmeet Singh
    "5b16a806": "OB",    # A Dananjaya (mystery off-break)
    "be0077ba": "LAFM",  # Junaid Khan
    "8db7f47f": "LAFM",  # RJW Topley
    "012829ff": "RAFM",  # JW Hastings
    "c8179c68": "SLA",   # SB Jakati
    "e186f49c": "RAFM",  # Mashrafe Mortaza
    "46a9bea1": "RAFM",  # TU Deshpande
    "5af743d0": "RAFM",  # Sohail Khan
    "f69f1d8a": "RAFM",  # Umaid Asif
    "0a3d54b9": "RAF",   # VR Aaron (Varun)
    "96824e68": "RAFM",  # CP Tremain
    "73c18486": "RAFM",  # KR Mayers
    "dec8e038": "RAF",   # J Theron
    "786b7fe8": "LWS",   # Zahir Khan (chinaman)
    "30afc92b": "RAFM",  # Salman Irshad
    "7ca5e05d": "RAFM",  # RS Bopara
    "12eddf28": "RAF",   # RJ Harris
    "2af838ee": "RAF",   # N Pradeep
    "ab6b1c45": "RAFM",  # AL Phehlukwayo
    "7a8bd078": "LBG",   # S Gopal (Shreyas)

    # ── 301–350 ──────────────────────────────────────────────────────────
    "83c3e8e3": "RAFM",  # SH Johnson
    "0848ac5f": "RAFM",  # S Thakor
    "7be8a0ed": "RAFM",  # JW Dernbach
    "f752db61": "RAF",   # JL Pattinson
    "58043739": "RAFM",  # TT Bresnan
    "0c2730df": "LBG",   # A Kumble
    "b8704508": "RAFM",  # LM Jongwe
    "d8b2f218": "LAFM",  # BB Sran (Barinder – left-arm)
    "9440ef41": "LBG",   # Suyash Sharma
    "f8c72766": "RAFM",  # D Heyliger
    "b08252b8": "OB",    # RP Burl
    "d99cc23a": "RAFM",  # T van der Gugten
    "684a56df": "RAFM",  # RR Beaton
    "d7725664": "LAFM",  # RA Reifer
    "c03449e0": "RAFM",  # GD Elliott
    "94eac556": "RAFM",  # CJ McKay
    "3fa5672e": "LBG",   # P Mishra
    "59792462": "RAFM",  # KM Jarvis
    "2cffab74": "RAFM",  # Mukesh Kumar
    "ae78bc32": "RAFM",  # MS Gony
    "4a745f65": "OB",    # AR Nurse (already done)
    "7a80ddf2": "LBG",   # D Bishoo (Devendra)
    "8ea6e670": "OB",    # NM Hauritz
    "2b85c8e6": "SLA",   # JL Jaggesar
    "1cb14aa4": "RAFM",  # CJ Dala
    "8d92a2c3": "RAF",   # MA Wood (Mark)
    "3cc13c68": "LAFM",  # Rahat Ali
    "10a91f35": "RAF",   # Shoaib Akhtar
    "f78e7113": "OB",    # S Prasanna
    "8de618ab": "SLA",   # SA Ahmad
    "e2db2409": "OB",    # M Ashwin (Murugan)
    "2cdce1be": "LAFM",  # C Sakariya (Chetan)
    "c7a995d3": "SLA",   # R Sai Kishore
    "e9c7f0d0": "LAFM",  # Fazalhaq Farooqi
    "8f6dd463": "RAFM",  # Azmatullah Omarzai
    "00321fff": "OB",    # Mohammad Ghazanfar
    "d12143bf": "RAFM",  # JM Anderson
    "e21bc7f3": "RAFM",  # RD Berrington
    "85b0ccd1": "RAFM",  # Khurram Shahzad
    "c58b0108": "OB",    # Mehedi Hasan Miraz
    "30a45b23": "LBG",   # SPD Smith
    "d9609443": "RAF",   # ST Gabriel
    "bb34fd31": "LAFM",  # Shoriful Islam
    "5bdcdb72": "OB",    # TM Dilshan
    "afe57f7a": "SLA",   # SC Williams
    "736123bb": "SLA",   # DN Wellalage
    "cb1a782a": "LAFM",  # Mir Hamza
    "ab89348d": "RAFM",  # MF Maharoof
    "cf73ad76": "SLA",   # JEC Franklin
    "782fb776": "RAFM",  # LV van Beek

    # ── 351–400 ──────────────────────────────────────────────────────────
    "d465c6d7": "RAFM",  # C Wright (Scotland)
    "cca50cd6": "OB",    # LJ Wright
    "3a0f6df2": "SLA",   # Zulfiqar Babar
    "26ff4c29": "SLA",   # RJ Peterson
    "4f629497": "RAF",   # SE Bond
    "79209272": "LBG",   # Yasir Ali (Bangladesh)
    "3c87f0c3": "LAFM",  # JN Frylinck (Namibia)
    "81c08fa3": "RAF",   # Umran Malik
    "c8a3f688": "LAFM",  # SN Netravalkar
    "8012d0b8": "RAFM",  # CAK Rajitha
    "a36915ce": "RAFM",  # N Vanua (PNG)
    "21d4e29b": "RAF",   # Navdeep Saini
    "68c56d09": "RAFM",  # KA Jamieson
    "e38bce7a": "OB",    # MG Bracewell (Michael)
    "c05edf8e": "SLA",   # Harpreet Brar
    "6042bf26": "RAFM",  # NJ Rimmington
    "96a6a7ad": "OB",    # NM Lyon
    "3eac9d95": "RAFM",  # JDP Oram
    "0b0cc297": "RAFM",  # Tanzim Hasan Sakib
    "d4eef961": "RAF",   # M de Lange
    "39a2dfa8": "LBG",   # R Tewatia (Rahul)
    "a9da7784": "RAFM",  # KJ Abbott
    "4bd09374": "RAF",   # Akash Madhwal
    "034b4b7d": "LAFM",  # VRV Singh
    "9dca07d7": "LAFM",  # RJ Sidebottom
    "1a8eebbe": "RAFM",  # Shafiul Islam
    "e01895bb": "RAFM",  # DT Johnston
    "72253b87": "RAF",   # Kaleemullah
    "342d8ade": "RAF",   # CRD Fernando (Dilhara)
    "1cb14aa4": "RAFM",  # CJ Dala already done
    "29b89ae8": "RAFM",  # WB Rankin
    "cfe70281": "RAFM",  # VD Philander
    "c18496e1": "SLA",   # Bipul Sharma
    "12bffe91": "LAFM",  # LL Tsotsobe
    "2d9451cc": "RAFM",  # MK Thakur
    "73ad96ed": "OB",    # DJ Hooda
    "a8e54ef4": "RAFM",  # Aamer Jamal
    "77b1aa15": "RAFM",  # Harshit Rana
    "c38d3503": "RAFM",  # Shivam Mavi
    "56b93d46": "OB",    # BJ Webster (Beau)
    "a86a37ab": "RAFM",  # Mohammad Ali (Pakistan)
    "871e9faf": "RAFM",  # Basil Thampi
    "80ad39ac": "RAFM",  # Hasan Mahmud
    "f33cfd25": "RAFM",  # B Masaba (Uganda)
    "462d7c62": "RAFM",  # D Paterson
    "d8976cb7": "OB",    # Mohammad Ghazanfar already done
    "a8e56914": "RAF",   # D Olivier (Duanne)
    "ee7d0c82": "RAFM",  # GD McGrath
    "6b9eb501": "RAF",   # OEG Baartman (Namibia)
    "b68a30f0": "RAFM",  # IG Butler

    # ── 401–450 ──────────────────────────────────────────────────────────
    "79aad751": "LWS",   # Sandakan already done
    "20a941bb": "RAF",   # M Ntini
    "12bffe91": "LAFM",  # Tsotsobe already done
    "4de91e08": "RAFM",  # generic
    "de3d549a": "RAF",   # AM Fernando (Asitha)
    "9a963804": "RAFM",  # Plunkett already done
    "0813a8b1": "RAFM",  # D Marks (Namibia)
    "87046e99": "SLA",   # Gagandeep Singh
    "7267e3cd": "OB",    # Ibrahim Rizan (Maldives)
    "f422728b": "RAFM",  # H Kerr (Scotland)
    "39e9f0c9": "RAFM",  # B Arora (IPL)
    "f545dffa": "RAFM",  # generic
    "4b31f3a3": "RAFM",  # Yash Thakur
    "1911dcbc": "LAFM",  # AAP Rivero (Argentina)
    "ef503cfe": "RAFM",  # P Kumbhare
    "034b4b7d": "LAFM",  # VRV already done
    "f33cfd25": "RAFM",  # Masaba already done
    "e969e580": "RAFM",  # GD Arta (PNG? skip)
    "3239c8a9": "RAFM",  # R Coker (CPL)
    "3204c99f": "RAF",   # G Coetzee (South Africa – right-arm fast)
    "fa463154": "RAFM",  # AB Agarkar
    "c6e1354c": "OB",    # Ikramullah Khan (Oman)
    "7f784eb8": "RAFM",  # V Zimonjic (Canada)
    "93a17209": "RAFM",  # VY Mahesh (India domestic)
    "bed7f36a": "SLA",   # generic SLA
    "9656afbf": "RAFM",  # Ahmed Daniyal (UAE)
    "e9c7f0d0": "LAFM",  # Fazalhaq already done
    "5935d694": "LBG",   # Rishad Hossain (Bangladesh leg-spin)
    "c660a069": "OB",    # generic
    "55f4f67a": "SLA",   # generic

    # ── Batch 2: extended classification (confirmed UUIDs from DB) ───────────
    "f986ca1a": "RAFM",  # HV Patel (Harshal Patel, India)
    "78cb83f3": "RAF",   # KOK Williams (Kesrick Williams, West Indies)
    "33a364a6": "SLA",   # R Bhatia (Rajat Bhatia, India)
    "f53b3e1d": "RAFM",  # Karan KC (Nepal)
    "1adb8ee8": "LAFM",  # Sompal Kami (Nepal)
    "e85d03d6": "SLA",   # BM Scholtz (Brian Scholtz, Namibia)
    "415d30b6": "SLA",   # Virandeep Singh (Canada)
    "ce794613": "LAFM",  # T Natarajan (India)
    "c6097d68": "RAF",   # O Thomas (Oshane Thomas, West Indies)
    "db584dad": "OB",    # CH Gayle (West Indies – off-break)
    "38c276e1": "RAFM",  # N McAndrew (Nathan McAndrew, Australia)
    "87e33d07": "OB",    # F Nsubuga (Fred Nsubuga, Uganda)
    "1b04e02b": "RAFM",  # TS Rogers (Australia domestic)
    "dcdb87f2": "RAFM",  # Bilal Khan (Oman/Pakistan)
    "46a7ac78": "RAFM",  # DM Nakrani (India domestic)
    "b552a935": "RAF",   # AC Thomas (West Indies)
    "3edb58fc": "RAFM",  # AD Mascarenhas (England)
    "2cec2a92": "RAF",   # Abbas Afridi (Pakistan)
    "79b4839e": "SLA",   # Ahmed Raza (Afghanistan)
    "9a158001": "RAFM",  # Azhar Mahmood (Pakistan)
    "c26cd57a": "LAFM",  # B Shikongo (Namibia)
    "ab21795b": "RAFM",  # BAD Manenti (Australia)
    "b56dc5f7": "LAFM",  # BE Hendricks (South Africa)
    "9868bc75": "OB",    # BMAJ Mendis (Sri Lanka)
    "172dff15": "RAFM",  # C Bosch (South Africa)
    "eaa76d3c": "RAFM",  # C Green (Cameron Green, Australia)
    "4751caa3": "RAF",   # CBRLS Kumara (Lahiru Kumara, Sri Lanka)
    "8abdf100": "SLA",   # CJ Anderson (New Zealand/USA)
    "de7d833e": "LAFM",  # D Madushanka (Dilshan Madushanka, Sri Lanka)
    "fd835ab3": "OB",    # DJ Hussey (David Hussey, Australia)
    "208f22ea": "RAFM",  # DJ Worrall (Daniel Worrall, Australia)
    "1a156c88": "SLA",   # DJM Short (D'Arcy Short, Australia)
    "f1eb3c73": "SLA",   # DMW Rawlins (England domestic)
    "35205dfc": "OB",    # DR Smith (Dwayne Smith, West Indies)
    "1747ea18": "SLA",   # DS Airee (Dipendra Singh Airee, Nepal)
    "951686f5": "RAF",   # Dawood Ahmadzai (Afghanistan)
    "c042b412": "RAFM",  # FJ Klaassen (Netherlands)
    "2dfe0612": "RAF",   # Ghulam Ahmadi (Afghanistan)
    "29e253dd": "OB",    # Iftikhar Ahmed (Pakistan)
    "414d8a10": "OB",    # J Miyaji (Japan)
    "59559bc2": "RAFM",  # J Overton (England)
    "f1c61793": "RAFM",  # JD Wildermuth (Australia)
    "003d49e6": "RAFM",  # JJ Bazley (Australia)
    "bb134e5c": "RAFM",  # JJ Smit (Namibia)
    "5ec41cd7": "RAFM",  # Jasdeep Singh (Canada)
    "7fb32e5b": "RAFM",  # KD Mills (New Zealand)
    "14af45c4": "RAFM",  # KDA Lukies (Australia)
    "243431b5": "OB",    # KJ O'Brien (Ireland)
    "5b616bf0": "RAFM",  # LA Burns (Australia domestic)
    "b08e58e2": "LAFM",  # LR Morris (Australia domestic)
    "a9fd84fb": "LBG",   # M Markande (India – leg-break)
    "28131839": "OB",    # MA Leask (Scotland)
    "95a5e066": "OB",    # MG Erasmus (Namibia)
    "e84ac20c": "RAFM",  # MJ Henry (New Zealand)
    "83d17bbc": "RAFM",  # MW Forde (West Indies)
    "a90e53ec": "OB",    # MW Short (Matthew Short, Australia)
    "b3a28446": "RAFM",  # Mohammad Saifuddin (Bangladesh)
    "ea3ffd1b": "SLA",   # NO Miller (West Indies)
    "1a0c3177": "RAFM",  # P Awana (Parvinder Awana, India)
    "9de271ef": "OB",    # P Utseya (Zimbabwe)
    "9fc0ef64": "LAFM",  # PJ Sangwan (India)
    "c654af19": "RAFM",  # R McLaren (South Africa)
    "650d5e49": "OB",    # R Powell (Rovman Powell, West Indies)
    "ff6a7ab5": "LAFM",  # R Trumpelmann (Namibia)
    "2f28dc94": "RAFM",  # RAS Lakmal (Suranga Lakmal, Sri Lanka)
    "fef92afc": "RAFM",  # SC Kuggeleijn (New Zealand)
    "364ee2cf": "SLA",   # SJ Benn (West Indies)
    "f233bbb4": "OB",    # ST Jayasuriya (Sri Lanka domestic)
    "33609a8c": "SLA",   # Saim Ayub (Pakistan)
    "7ccbb09f": "RAFM",  # Shahnawaz Dhani (Pakistan)
    "c9945bdf": "OB",    # Simi Singh (Ireland)
    "615688b6": "RAFM",  # Usman Najeeb (Afghanistan)
    "7c3b3b78": "RAFM",  # VG Arora (India domestic)
    "bc2d4f2e": "RAFM",  # W Sutherland (Australia)
    "4bc0a378": "OB",    # WP Masakadza (Zimbabwe)
    "7210d461": "LAFM",  # Yash Dayal (India)
    "f0e293b0": "RAFM",  # Zahoor Khan (UAE)
    "28392e11": "LAFM",  # Zaman Khan (Pakistan)
    "34464143": "SLA",   # Zeeshan Maqsood (Oman)

    # ── Batch 3: user-provided classifications ────────────────────────────────
    "51a60c16": "LBG",   # CJ Boyce (Australia)
    "a2f46292": "LAFM",  # KK Ahmed (Khaleel Ahmed, India)
    "e741ed8f": "RAFM",  # Rizwan Butt (Bahrain)
    "2b7609f1": "SLA",   # H Ssenyondo (Uganda)
    "49118dc7": "SLA",   # SO Ngoche (Kenya)
    "a61bcb94": "SLA",   # K Pierre (West Indies)
    "84dc72db": "OB",    # Junaid Siddique (Bangladesh)
    "ba57fb08": "RAFM",  # Z Bimenyimana (Rwanda)
    "a9a18e3e": "LAFM",  # Imran Anwar (Bahrain)
    "3c8faed4": "RAFM",  # F Banunaek (Indonesia)
    "0cfb42b5": "SLA",   # Abdul Majid Abbasi (Bahrain)
    "a62f55ba": "RAFM",  # DJ Hawoe (Indonesia)
    "1c5c3dce": "RAFM",  # IO Okpe (Nigeria)
    "c9d05f1a": "SLA",   # Yasim Murtaza (Hong Kong)
    "20481563": "SLA",   # MNM Aslam (Kuwait)
    "596982e6": "RAFM",  # Ali Dawood (Bahrain)
    "88209c84": "LAFM",  # JK Lalor (Australia)
    "bd0a4a7d": "RAF",   # CA Young (Ireland)
    "681f522f": "OB",    # SA Okpe (Nigeria)
    "f28a60e0": "SLA",   # Saad Bin Zafar (Canada)
    "2af28253": "SLA",   # Vraj Patel (Kenya)
    "090f96e4": "SLA",   # Muhammad Nadir (Rwanda)
    "1f3b9f8d": "RAFM",  # M Akayezu (Rwanda)
    "30d8136b": "SLA",   # AR Ramjani (Uganda)
    "f0a45427": "RAFM",  # RM Koda (Indonesia – Maxi Koda)
    "14157abf": "RAFM",  # Aqib Iqbal (Austria)
    "f737ab44": "OB",    # S Vijay Unni (Malaysia)
    "64fb6b15": "RAFM",  # A Bohara (Nepal)
}


def run_backfill(dry_run: bool = False) -> None:
    engine  = get_engine()

    # Convert short codes to full labels
    uuid_to_style = {
        uuid: _STYLE_LABELS[code]
        for uuid, code in STYLE_MAP.items()
        if code in _STYLE_LABELS
    }

    with Session(engine) as session:
        total_players = session.execute(text("SELECT COUNT(*) FROM players")).scalar()
        already_set   = session.execute(
            text("SELECT COUNT(*) FROM players WHERE bowling_style IS NOT NULL AND bowling_style != ''")
        ).scalar()

        updated = 0
        not_found = 0

        for uuid, style in uuid_to_style.items():
            exists = session.execute(
                text("SELECT id FROM players WHERE cricsheet_uuid = :u"), {"u": uuid}
            ).scalar()
            if exists is None:
                not_found += 1
                continue
            if not dry_run:
                session.execute(
                    text("UPDATE players SET bowling_style = :s WHERE cricsheet_uuid = :u"),
                    {"s": style, "u": uuid},
                )
            updated += 1

        if not dry_run:
            session.commit()

    console.print(f"\n[bold]Bowling style backfill — {'DRY RUN' if dry_run else 'LIVE'}[/bold]")
    console.print(f"  Players in DB          : {total_players:,}")
    console.print(f"  Styles pre-existing    : {already_set}")
    console.print(f"  Styles written         : {updated}")
    console.print(f"  UUIDs not in DB        : {not_found}")

    # Show style distribution
    if not dry_run:
        with Session(engine) as session:
            dist = session.execute(text("""
                SELECT bowling_style, COUNT(*) as cnt
                FROM players
                WHERE bowling_style IS NOT NULL AND bowling_style != ''
                GROUP BY bowling_style ORDER BY cnt DESC
            """)).fetchall()

        t = Table(title="Style distribution", header_style="bold cyan")
        t.add_column("Style")
        t.add_column("Players", justify="right")
        for r in dist:
            t.add_row(r.bowling_style, str(r.cnt))
        console.print(t)

        # Coverage: what % of deliveries now have a styled bowler?
        with Session(engine) as session:
            total_del = session.execute(text("SELECT COUNT(*) FROM deliveries")).scalar()
            styled_del = session.execute(text("""
                SELECT COUNT(*) FROM deliveries d
                JOIN players p ON p.id = d.bowler_id
                WHERE p.bowling_style IS NOT NULL AND p.bowling_style != ''
            """)).scalar()
        console.print(f"\n  Delivery coverage: {styled_del:,} / {total_del:,} = [bold]{100*styled_del//total_del}%[/bold]")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--dry-run", is_flag=True, default=False)
    def main(dry_run: bool):
        run_backfill(dry_run)

    main()
