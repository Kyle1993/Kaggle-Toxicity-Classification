import os,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle
import gc
gc.enable()
import time
from tqdm import tqdm,tqdm_notebook
import collections
import shutil
import re
from nltk.tokenize import TweetTokenizer
from keras.preprocessing import text, sequence
import emoji
from datetime import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader,TensorDataset,Dataset
torch.cuda.set_device(0)

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

package_dir = "../input/pp-bert"
sys.path.insert(0, package_dir)
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam

import warnings
warnings.filterwarnings('ignore')


##############################################################
#                      Text Process                          #
##############################################################

class Text_Process():
    def __init__(self):
        self.contraction_dict = {" Im ": " I am ", " Padmaavat ": " Padmavati ", " Padmaavati ": " Padmavati ",
                                 " Padmavat ": " Padmavati ", " S.P ": " ", " U.S ": " USA ", " U.s ": "USA",
                                 " coinvest ": " invest ", " documentarie ": " documentaries ", " don t ": " do not ",
                                 " dont ": " do not ", " dsire ": " desire ", " govermen ": "goverment",
                                 " hinus ": " hindus ", " incect ": " insect ",
                                 " internshala ": " internship and online training platform in India ",
                                 " jusification ": "justification", " justificatio ": "justification",
                                 " padmavat ": " Padmavati ", " racious ": "discrimination expression of racism",
                                 " righten ": " tighten ", " s.p ": " ", " u ": " you ", " u r ": " you are ",
                                 " u.S ": " USA ", " u.k ": " UK ", " u.s ": " USA ", " yr old ": " years old ",
                                 "#": "", "#metoo": "MeToo",
                                 "'cause": "because", "...": ".",
                                 "23 andme": "privately held personal genomics and biotechnology company in California",
                                 "2fifth": "twenty fifth", "2fourth": "twenty fourth", "2nineth": "twenty nineth",
                                 "2third": "twenty third", "4fifth": "forty fifth", "4fourth": "forty fourth",
                                 "4sixth": "forty sixth",
                                 "@ usafmonitor": "", "A**": "ass", "A****": "assho", "ABnegetive": "Abnegative",
                                 "AIN'T": "is not", "ARE'NT": "are not", "AREN'T": "are not",
                                 "AddMovement": "Add Movement", "AdsTargets": "Ads Targets", "Adsresses": "address",
                                 "Afircans": "Africans", "Agoraki": "Ago raki", "Ahemadabad": "Ahmadabad",
                                 "Airindia": "Air india", "Alamotyrannus": "Alamo tyrannus", "Allmang": "All mang",
                                 "AlwaysHired": "Always Hired", "Amedabad": "Ahmedabad",
                                 "AmenityCompany": "Amenity Company", "Amharc": "Amarc",
                                 "Amigofoods": "Amigo foods", "Amn't": "is not", "Anglosaxophone": "Anglo saxophone",
                                 "Anonoymous": "Anonymous", "Antergos": "Anteros", "Antritrust": "Antitrust",
                                 "Apartmentfinder": "Apartment finder", "ArangoDB": "Arango DB", "Are'nt": "are not",
                                 "Aremenian": "Armenian",
                                 "Aren't": "are not", "Armenized": "Armenize", "Arn't": "are not", "Ashifa": "Asifa",
                                 "Asianisation": "Becoming Asia", "Asiasoid": "Asian", "Audetteknown": "Audette known",
                                 "Augumented": "Augmented", "Auragabad": "Aurangabad", "Austertana": "Auster tana",
                                 "Austira": "Australia", "Australia9": "Australian", "Australianism": "Australian ism",
                                 "Australua": "Australia", "Austrelia": "Australia", "Austrialians": "Australians",
                                 "Austrolia": "Australia", "Auwe": "wonder", "Ayahusca": "Ayahausca",
                                 "Babadook": "a horror drama film",
                                 "Badaganadu": "Brahmin community that mainly reside in Karnataka",
                                 "Badonicus": "Sardonicus", "Ballyrumpus": "Bally rumpus",
                                 "Baloochistan": "Balochistan", "Banggood": "Bang good",
                                 "BasedStickman": "Based Stickman", "Beerus": "the God of Destruction",
                                 "Beluchistan": "Balochistan", "Betterindia": "Better india", "Bhagwanti": "Bhagwant i",
                                 "Bhraman": "Braman", "Bhushi": "Thushi", "Bindusar": "Bind usar",
                                 "BingoBox": "an investment company", "Blackphone": "Black phone",
                                 "Blackphones": "Black phones", "Bogosort": "Bogo sort", "Bonechiller": "Bone chiller",
                                 "Book'em": "book them", "Boothworld": "Booth world",
                                 "Borokabama": "Barack Obama", "Boruto": "Naruto Next Generations",
                                 "Brahmanwad": "Brahman wad", "Brahmanwadi": "Brahman wadi", "Brainsway": "Brains way",
                                 "Bramanical": "Brahmanical", "Brexit": "british exit", "Brightspace": "Brights pace",
                                 "Brusssels": "Brussels", "BudgetAir": "Budget Air",
                                 "Bugetti": "Bugatti", "Busscas": "Buscas", "C'Mon": "come on",
                                 "C'mooooooon": "come on", "CANN'T": "can not", "CON'T": "can not",
                                 "CONartist": "con-artist", "CONgressi": "Congress", "COUD'VE": "could have",
                                 "COULD'VE": "could have",
                                 "COx": "cox", "Cab't": "can not", "Calculus1": "Calculus", "Caligornia": "California",
                                 "Can't": "can not", "Cant't": "can not",
                                 "CareOnGo": "India first and largest Online distributor of medicines",
                                 "Catallus": "Catullus", "Causians": "Crusians", "Chickengonia": "Chicken gonia",
                                 "Chromecast": "Chrome cast", "Chronexia": "Chronaxia", "CityAirbus": "City Airbus",
                                 "Clearworld": "Clear world", "Cloudways": "Cloud ways", "CodeAgon": "Code Agon",
                                 "CommentSafe": "Comment Safe", "Commissionerates": "Commissioner ates",
                                 "Con't": "can not", "Confundus": "Con fundus",
                                 "Congoid": "Congolese", "ConsciousX5": "conscious",
                                 "Cooldige": "the 30th president of the United States", "Coolmuster": "Cool muster",
                                 "Corypheus": "Coryphees", "Could'nt": "could not", "Couldn't": "could not",
                                 "Cuckerberg": "Zuckerberg", "CusJo": "Cusco", "Customzation": "Customization",
                                 "DIdn't": "did not", "DOES'NT": "does not", "DOESEN'T": "does not",
                                 "DOESN'T": "does not", "DON'T": "do not", "DONT'T": "do not", "DOesn't": "does not",
                                 "Dardandus": "Dardanus", "Ddn't": "did not", "Degenerous": "De generous",
                                 "Democratizationed": "Democratization ed", "Demogorgan": "Demogorgon",
                                 "Demonetisation": "demonetization", "Deshabhimani": "Desha bhimani",
                                 "Designbold": "Online Photo Editor Design Studio", "Devendale": "Evendale",
                                 "DiDn't": "did not", "Didn't": "did not", "Didn't_work": "did not work",
                                 "Didn`t": "did not",
                                 "Dn't": "do not", "Do'nt": "do not", "Does'nt": "does not", "Does't": "does not",
                                 "Doesn't": "does not", "Don''t": "do not", "Don'TCare": "do not care",
                                 "Don't": "do not", "Don'ts": "do not", "DoneClaim": "Done Claim",
                                 "DoneDone": "Done Done", "Dont't": "do not", "Dormmanu": "Dormant",
                                 "Dose't": "does not", "Dowsn't": "does not", "Dpn't": "do not",
                                 "Draconius": "Draconis", "Dragonchain": "Dragon chain",
                                 "Dragonkeeper": "Dragon keeper", "Dragonknight": "Dragon knight",
                                 "Dramafever": "Drama fever", "Dream11": " fantasy sports platform in India ",
                                 "Drumpf": "trump", "Dushasana": "Dush asana", "Dusra": "Dura", "Ecoworld": "Eco world",
                                 "Electroneum": "Electro neum", "Elitmus": "eLitmus", "Emmanvel": "Emmarvel",
                                 "Emouluments": "Emoluments",
                                 "Empericus": "Imperious", "Emplement": "Implement", "Employmment": "Employment",
                                 "Enchacement": "Enchancement", "Enchroachment": "Encroachment",
                                 "Encironmental": "Environmental", "EndFragment": "End Fragment",
                                 "Entwicklungsroman": "Entwicklungs roman", "Epicodus": "Episodes",
                                 "EssayTyper": "Essay Typer",
                                 "Eurooe": "Europe", "Ev'rybody": "everybody", "EveryoneDiesTM": "EveryoneDies TM",
                                 "F**": "fuc", "F***": "fuck", "F**K": "fuck", "F**k": "fuck", "FAMINAZIS": "FEMINAZIS",
                                 "FacePlusPlus": "Face PlusPlus", "FakeNews": "fake news",
                                 "Fanessay": "Fan essay", "Fantocone": "Fantocine",
                                 "Farmhousebistro": "Farmhouse bistro", "Fernbus": "Fern bus", "Feymann": "Heymann",
                                 "Firdausiya": "Firdausi ya", "Flixbus": "Flix bus", "Français": "France",
                                 "Freemansonry": "Freemasonry", "Freshersworld": "Freshers world",
                                 "Freus": "Frees", "Frosione": "Erosion", "Fuckboy": "fuckboy", "Fuckboys": "fuckboy",
                                 "Fundamantal": "fundamental", "Fusanosuke": "Fu sanosuke",
                                 "FusionDrive": "Fusion Drive", "FusionGPS": "Fusion GPS", "Führer": "Fuhrer",
                                 "G'Night": "goodnight",
                                 "G'bye": "goodbye", "G'morning": "good morning", "G'night": "goodnight",
                                 "GAPbuster": "GAP buster", "GadgetPack": "Gadget Pack", "Galastop": "Galas top",
                                 "Gamenights": "Game nights", "Garthago": "Carthago", "Gaudry - Schost": "",
                                 "Geev'um": "give them",
                                 "Geftman": "Gentman", "Genderfluid": "Gender fluid", "Germanised": "German ised",
                                 "Germanity": "German", "Germanyl": "Germany l", "Ghumendra": "Bhupendra",
                                 "Give'em": "give them", "Gobackmodi": "Goback modi",
                                 "GoldenDict": "open-source dictionary program",
                                 "Goldmont": "microarchitecture in Intel",
                                 "Grab'em": "grab them", "Greenseer": "people who possess the magical ability",
                                 "Guardtime": "Guard time", "Gujjus": "derogatory Gujarati", "GusDur": "Gus Dur",
                                 "Gusenberg": "Gutenberg", "Gyroglove": "wearable technology", "Gʀᴇat": "great",
                                 "H***": "hole", "HADN'T": "had not",
                                 "HASN'T": "has not", "HAVEN'T": "have not", "HackerRank": "Hacker Rank",
                                 "Haden't": "had not", "Haffmann": "Hoffmann", "Hanfus": "Hannus",
                                 "HasAnyone": "Has Anyone", "Hasidus": "Hasid us", "Haufman": "Kaufman",
                                 "Havn't": "have not",
                                 "He''s": "he is", "Hellochinese": "Hello chinese", "Hexanone": "Hexa none",
                                 "Hhhow": "how", "Highschoold": "High school", "Hindustanis": "",
                                 "Hindusthanis": "Hindustanis", "Hold'um": "hold them", "Holocoust": "Holocaust",
                                 "Honeyfund": "Honey fund",
                                 "Hopstop": "Hops top", "Hotstar": "Hot star", "Howddo": "How do", "Howeber": "However",
                                 "Howknow": "Howk now", "Howlikely": "How likely", "Howmaney": "How maney",
                                 "Howwould": "How would", "Hue Mungus": "feminist bait", "Hugh Mungus": "feminist bait",
                                 "Humanpark": "Human park", "Huskystar": "Hu skystar", "Husnai": "Hussar",
                                 "Huswifery": "Huswife ry", "Hwhat": "what", "Hydeabad": "Hyderabad",
                                 "Hypercubus": "Hypercubes", "I''ve": "I have", "I'D": "I would",
                                 "I'DVE": "I would have",
                                 "I'Il": "I will", "I'M": "I am", "I'd": "I would", "I'd've": "I would have",
                                 "I'don": "I do not", "I'l": "I will", "I'll": "I will", "I'll've": "I will have",
                                 "I'llbe": "I will be", "I'lll": "I will",
                                 "I'm": "I am", "I'ma": "I am a", "I'mm": "I am", "I'mma": "I am a", "I'v": "I have",
                                 "I've": "I have", "I'vemade": "I have made", "I'veposted": "I have posted",
                                 "I'veve": "I have", "I'vÈ": "I have",
                                 "I'Μ": "I am", "IJustDon'tThink": "I just do not think", "IPhone7": "Iphone",
                                 "ISN`T": "is not", "IT'D": "it would", "Icompus": "Corpus",
                                 "Idon'tgetitatall": "I do not get it at all", "Idonesia": "Indonesia",
                                 "Ignasius": "Ignacius", "Ilusha": "Ilesha",
                                 "Imaprtus": "Impetus", "Inatrumentation": "Instrumentation",
                                 "Incestious": "Incestuous", "Indigoflight": "Indigo flight", "Indominus": "In dominus",
                                 "Industrailzed": "Industrialized", "Indusyry": "Industry", "Ingenhousz": "Ingenious",
                                 "Inquoraing": "inquiring", "Insdians": "Indians",
                                 "Inspirus": "Inspires", "Instumentation": "Instrumentation", "Intuous": "Virtuous",
                                 "Invetsment": "Investment", "Inviromental": "Environmental", "Iovercome": "I overcome",
                                 "Is't": "is not", "Isdhanbad": "Is dhanbad", "Isn't": "is not", "Istop": "I stop",
                                 "It''s": "it is", "It'also": "it is also", "It`s": "it is", "Itylus": "I tylus",
                                 "Iv'e": "I have", "Ivanious": "Avanious", "Janewright": "Jane wright",
                                 "JanusGraph": "Janus Graph", "Jarrus": "Harrus", "Jeisus": "Jesus",
                                 "Jerusalsem": "Jerusalem", "Jeruselam": "Jerusalem", "Jigolo": "Gigolo",
                                 "Jlius": "Julius", "Jobberman": "Jobber man", "Joboutlooks": "Job outlooks",
                                 "Josephius": "Josephus", "Juilus": "Julius", "JustBasic": "Just Basic",
                                 "JustForex": "Just Forex",
                                 "JustinKase": "Justin Kase", "Kalitake": "Kali take", "Kamikazis": "Kamikazes",
                                 "Karonese": "Karo people Indonesia", "Kashmiristan": "Kashmir",
                                 "Kaushika": "Kaushik a", "Khushali": "Khushal i", "Kick'em": "kick them",
                                 "Kim Jong-Un": "The president of North Korea", "Koncerned": "concerned",
                                 "Kouldn't": "could not", "Kousakis": "Kou sakis", "Koushika": "Koushik a",
                                 "Kousseri": "Kousser i", "Kremenchuh": "Kremenchug", "Kumarmangalam": "Kumar mangalam",
                                 "Kunstlerroman": "Kunstler roman", "LOCK'EM": "lock them", "LOCK'UM": "lock them",
                                 "Laakman": "Layman",
                                 "Lakhsman": "Lakhs man", "Languagetool": "Language tool",
                                 "Latinamericans": "Latin americans", "Le'ts": "let us",
                                 "LeewayHertz": "Blockchain Company", "Lensmaker": "Lens maker",
                                 "Light'em": "light them", "Lingayatism": "Lingayat", "Lllustrate": "Illustrate",
                                 "Look'em": "look them",
                                 "Love'em": "love them", "Loy Machedeo": "person",
                                 "Loy Machedo": " Motivational Speaker ", "Luscinus": "Luscious",
                                 "Luxemgourg": "Luxembourg", "Ma'am": "madam", "Magetta": "Maretta",
                                 "Maimonedes": "Maimonides", "Manamement": "Management", "MangoPay": "Mango Pay",
                                 "Mangolian": "Mongolian", "Manuscriptology": "Manuscript ology",
                                 "Marcusean": "Marcuse an", "Marimanga": "Mari manga", "Marrakush": "Marrakesh",
                                 "Massahusetts": "Massachusetts", "MasterColonel": "Master Colonel",
                                 "Mathusla": "Mathusala", "Maurititus": "Mauritius", "Megatapirus": "Mega tapirus",
                                 "MercadoPago": "Mercado Pago", "Mevius": "Medius", "Microservices": "Micro services",
                                 "Mifeprostone": "Mifepristone", "Milo Yianopolous": "a British polemicist",
                                 "Minangkabaus": "Minangkabau s", "Missuses": "Miss uses", "Mistworlds": "Mist worlds",
                                 "Moduslink": "Modus link", "Mogolia": "Mongolia",
                                 "Monegasques": "Monegasque s", "Moneyfront": "Money front",
                                 "MongoImport": "Mongo Import", "Montogo": "montego", "Moongot": "Moong ot",
                                 "Mouseflow": "Mouse flow", "Moushmee": "Mousmee", "Moussolini": "Mussolini",
                                 "MuscleBlaze": "Muscle Blaze", "Musevi": "the independence of Mexico",
                                 "Musharrf": "Musharraf", "Mushlims": "Muslims", "Musickers": "Musick ers",
                                 "Musigma": "Mu sigma", "Musino": "Musion", "Muslimophobe": "Muslim phobic",
                                 "Muslimophobia": "Muslim phobia", "Mustansiriyah": "Mustansiriya h",
                                 "Musturbation": "Masturbation", "Mutilitated": "Mutilated",
                                 "Nagamandala": "Naga mandala", "Namit Bathla": "Content Writer",
                                 "Nautlius": "Nautilus", "Nearbuy": "Nearby", "Needn't": "need not", "Nennus": "Genius",
                                 "Neulife": "Neu life", "Neurosemantics": "Neuro semantics", "Nibirus": "Nibiru",
                                 "Niggor": "black hip-hop and electronic artist",
                                 "Nobushi": "No bushi", "Nonchristians": "Non Christians", "Noonein": "Noo nein",
                                 "Nutrament": "Nutriment", "O'bamacare": "Obamacare", "OU'RE": "you are",
                                 "Obumblers": "bumblers", "Oligodendraglioma": "Oligodendroglioma",
                                 "Olnhausen": "Olshausen", "OnePlus": "Chinese smartphone manufacturer",
                                 "Oneplus": "OnePlus", "Outlook365": "Outlook 365", "P***": "porn", "P****": "pussy",
                                 "Pakistainies": "Pakistanis", "Pakustan": "Pakistan",
                                 "Pangoro": "cantankerous Pokemon", "Panromantic": "Pan romantic",
                                 "Pantherous": "Panther ous", "Parkistan": "Pakistan",
                                 "Parkistinian": "Pakistani", "Pasmanda": "Pas manda", "Pay'um": "pay them",
                                 "Pedogogical": "Pedological", "PerformanceTesting": "Performance Testing",
                                 "Perliament": "Parliament", "Ph.D": "PhD", "Phinneus": "Phineus",
                                 "Photoacoustics": "Photo acoustics", "Photofeeler": "Photo feeler",
                                 "Pick'em": "pick them", "Pictones": "Pict ones", "PlayMagnus": "Play Magnus",
                                 "PlayerUnknown": "Player Unknown", "Playerunknown": "Player unknown",
                                 "Pliosaurus": "Pliosaur us", "Pornosexuality": "Porno sexuality",
                                 "Priebuss": "Prie buss", "Promenient": "Provenient", "Prussophile": "Russophile",
                                 "Pulphus": "Pulpous", "Pushbullet": "Push bullet", "Pushkaram": "Pushkara m",
                                 "QMAS": "Quality Migrant Admission Scheme", "Qoura": "Quora", "REFERNECE": "REFERENCE",
                                 "RPatah - Tan Eng Hwan": "Silsilah", "Ra - apist": "rapist", "Ra apist": "Rapist",
                                 "Raddus": "Radius",
                                 "Radijus": "Radius", "Rahmanland": "Rahman land", "Rajsthan": "Rajasthan",
                                 "Rajsthanis": "Rajasthanis", "RangerMC": "car", "Rankholders": "Rank holders",
                                 "Rasayanam": "Rasayan am", "Redmi": "Xiaomi Mobile", "Reinfectus": "reinfect",
                                 "Remainers": "remainder",
                                 "Representment": "Rep resentment", "Retrocausality": "Retro causality",
                                 "Reveuse": "Reve use", "Rewardingways": "Rewarding ways", "Rheusus": "Rhesus",
                                 "Richmencupid": "rich men dating website", "Rigetti": "Ligetti",
                                 "Rimworld": "Rim world", "Ringostat": "Ringo stat",
                                 "Rivigo": "technology-enabled logistics company",
                                 "RolloverBox": "Rollover Box", "RomanceTale": "Romance Tale", "Romanium": "Romanum",
                                 "Rovman": "Roman", "Rumenova": "Rumen ova", "Run'em": "run them",
                                 "Russiagate": "Russia gate", "Russions": "Russians",
                                 "Russosphere": "russia sphere of influence", "Rustichello": "Rustic hello",
                                 "Rustyrose": "Rusty rose", "S**": "shi", "S***": "shit", "SB91": "senate bill",
                                 "SEND'EM": "send them", "SHE'LL": "she will", "SHOOT'UM": "shoot them",
                                 "SHouldn't": "should not", "SJWs": "social justice warrior", "SPEICIAL": "SPECIAL",
                                 "Sadhgurus": "Sadh gurus", "Saggittarius": "Sagittarius", "Sapiosexual": "sapiosexual",
                                 "Sapiosexuals": "sapiosexual", "Sarumans": "Sarum ans", "Sasabone": "Sasa bone",
                                 "Satannus": "Sat annus", "Sauskes": "Causes", "Savvius": "Savvies",
                                 "Sedataious": "Seditious",
                                 "Senousa": "Venous", "Setya Novanto": "a former Indonesian politician",
                                 "Sevenfriday": "Seven friday", "Sh**": "shit", "She''l": "she will",
                                 "Shoudn't": "should not", "Shouldn't": "should not", "Showh": "Show",
                                 "Shubman": "Subman", "Sigorn": "son of Styr",
                                 "Skillport": "Army e-Learning Program", "Skillselect": "Skills elect",
                                 "Sloatman": "Sloat man", "Softthinks": "Soft thinks", "Som'thin": "something",
                                 "Southindia": "South india", "SoyBoys": "cuck men lacking masculine characteristics",
                                 "Spectrastone": "Spectra stone", "Spoolman": "Spool man", "Sshouldn't": "should not",
                                 "StartFragment": "Start Fragment", "SteLouse": "Ste Louse",
                                 "Stegosauri": "stegosaurus", "Steymann": "Stedmann", "Stocklogos": "Stock logos",
                                 "Stonehart": "Stone hart", "Stonemen": "Stone men", "Straussianism": "Straussian ism",
                                 "Subramaniyan": "Subramani yan", "Sulamaniya": "Sulamani ya",
                                 "Sumaterans": "Sumatrans", "Suparwoman": "Superwoman", "Superowoman": "Superwoman",
                                 "Susanoomon": "Susanoo mon", "Susgaon": "Surgeon", "Sushena": "Saphena",
                                 "Sussia": "ancient Jewish village", "Swissgolden": "Swiss golden",
                                 "Syncway": "Sync way", "TFWs": "tuition fee waiver",
                                 "THAT'LL": "that will", "THEY'VE": "they have", "Tagushi": "Tagus hi",
                                 "Taharrush": "Tahar rush", "Take'em": "take them", "Tarumanagara": "Taruma nagara",
                                 "Tastaman": "Rastaman", "Techmakers": "Tech makers", "Technoindia": "Techno india",
                                 "Tell'em": "tell them",
                                 "Telloway": "Tello way", "Tennesseus": "Tennessee", "Terroristan": "terrorist",
                                 "TestoUltra": "male sexual enhancement supplement", "Tetherusd": "Tethered",
                                 "Thaedus": "Thaddus", "That''s": "that is", "Ther'es": "there is",
                                 "They'er": "they are", "They'l": "they will",
                                 "They'lll": "they will", "They_didn't": "they did not", "Theyr'e": "they are",
                                 "Theyv'e": "they have", "Tiannanmen": "Tiananmen", "Tiltbrush": "Tilt brush",
                                 "Tobagoans": "Tobago ans", "Touchtime": "Touch time",
                                 "TradeCommander": "Trade Commander", "Trampaphobia": "Trump aphobia",
                                 "Tridentinus": "mushroom", "Trimp": "Trump", "TrumpDoesn'tCare": "Trump does not care",
                                 "TrumpDon'tCareAct": "Trump do not care act", "TrumpIDin'tCare": "Trump did not care",
                                 "Trumpcare": "Trump health care system", "Trumpers": "president trump",
                                 "Trumpian": "viewpoints of President Donald Trump",
                                 "Trumpism": "philosophy and politics espoused by Donald Trump",
                                 "Trumpists": "admirer of Donald Trump",
                                 "Trumpster": "trumpeters", "Trumpsters": "Trump supporters", "TrustKit": "Trust Kit",
                                 "Trustclix": "Trust clix", "U'r": "you are", "U.K.": "UK", "U.S.": "USA",
                                 "U.S.A": "USA", "U.S.A.": "USA", "U.s.": "USA",
                                 "U.s.p": "", "UCSanDiego": "UC SanDiego", "USA''s": "USA",
                                 "USAgovernment": "USA government", "Unacademy": "educational technology company",
                                 "Understandment": "Understand ment", "Undertale": "video game", "Unglaus": "Ung laus",
                                 "Unitedstatesian": "United states", "Unwanted72": "Unwanted 72",
                                 "Upwork": "Up work", "Vancouever": "Vancouver", "Venus25": "Venus", "Vinis": "vinys",
                                 "Virushka": "Great Relationships Couple", "Vishnus": "Vishnu",
                                 "Vodafone2": "Vodafones", "Vote'em": "vote them", "W'ell": "we will",
                                 "WASN'T": "was not",
                                 "WAsn't": "was not", "WE'D": "we would", "WE'LL": "we will", "WE'RE": "we are",
                                 "WEREN'T": "were not", "WON'T": "will not", "WON't": "will not",
                                 "WOULD'NT": "would not", "WOULD'VE": "would have", "WW 1": " WW1 ",
                                 "WW 2": " WW2 ", "Wannaone": "Wanna one", "Washwoman": "Wash woman",
                                 "Wasn't": "was not", "We''ll": "we will", "We'll": "we will",
                                 "We'really": "we are really", "Wedgieman": "Wedgie man", "Wedugo": "Wedge",
                                 "Wenzeslaus": "Wenceslaus",
                                 "Weren't": "were not", "Wern't": "were not", "What sApp": "WhatsApp",
                                 "What's": "what is", "Whatcould": "What could", "Whateducation": "What education",
                                 "Whatevidence": "What evidence", "Whatmakes": "What makes", "Whatwould": "What would",
                                 "Whichcountry": "Which country",
                                 "Whichtreatment": "Which treatment", "Who''s": "who is",
                                 "Whoimplemented": "Who implemented", "Whwhat": "What", "Whybis": "Why is",
                                 "Whyco-education": "Why co-education", "Wildstone": "Wilds tone", "Williby": "will by",
                                 "Willowmagic": "Willow magic", "WillsEye": "Will Eye",
                                 "Withgott": "Without", "Womansplaining": "feminist", "Won'tdo": "will not do",
                                 "WorkFusion": "Work Fusion", "Worldkillers": "World killers", "Would't": "would not",
                                 "Wouldn'T": "would not", "Wouldn't": "would not", "Y'ALL": "you all",
                                 "Y'All": "you all",
                                 "Y'all": "you all", "Y'know": "you know", "YOU'RE": "you are", "YOU'VE": "you have",
                                 "YOUR'E": "you are", "Yahtzees": "Yahtzee", "Yegorovich": "Yegorov ich",
                                 "Yo'ure": "you are", "You''re": "you are", "You'all": "you all",
                                 "You'ld": "you would", "You'reOnYourOwnCare": "you are on your own care",
                                 "You'res": "you are", "You'rethinking": "you are thinking", "You'very": "you are very",
                                 "Your'e": "you are", "Yousician": "Musician", "Yoyou": "you",
                                 "Yuguslavia": "Yugoslavia", "Yumstone": "Yum stone",
                                 "Yutyrannus": "Yu tyrannus", "Yᴏᴜ": "you", "Zamusu": "Amuse", "ZenFone": "Zen Fone",
                                 "Zenfone": "Zen fone", "Zhuchengtyrannus": "Zhucheng tyrannus", "a**": "ass",
                                 "a****": "assho", "abandined": "abandoned", "abandonee": "abandon",
                                 "abcmouse": "abc mouse", "abdimen": "abdomen", "abhimana": "abhiman a",
                                 "abonymously": "anonymously", "abussive": "abusive", "accedentitly": "accidentally",
                                 "accomany": "accompany", "accompishments": "accomplishments",
                                 "accomplihsments": "accomplishments", "accomplishmments": "accomplishments",
                                 "accompliushments": "accomplishments", "accomploshments": "accomplishments",
                                 "accomplsihments": "accomplishments", "accompplishments": "accomplishments",
                                 "accusition": "accusation", "achecomes": "ache comes", "achiements": "achievements",
                                 "achievership": "achievers hip", "achivenment": "achievement",
                                 "achviements": "achievements",
                                 "acidtone": "acid tone", "activationenergy": "activation energy",
                                 "activemoney": "active money", "acumens": "acumen s", "adgestment": "adjustment",
                                 "adhaar": "Adhara", "advamcements": "advancements", "advantegeous": "advantageous",
                                 "aesexual": "asexual", "aevelopment": "development",
                                 "africaget": "africa get", "afterafterlife": "after afterlife",
                                 "afvertisements": "advertisements", "aggeement": "agreement",
                                 "agreementindia": "agreement india", "ahemadabad": "Ahmadabad",
                                 "ahmdabad": "Ahmadabad", "ahmedbad": "Ahmedabad", "ain't": "is not",
                                 "airindia": "air india",
                                 "aknowlege": "knowledge", "alahabad": "Allahabad", "alggorithms": "algorithms",
                                 "algorithimic": "algorithmic", "algorithom": "algorithm", "algoritmic": "algorismic",
                                 "algorthmic": "algorithmic", "algortihms": "algorithms", "alimoney": "alimony",
                                 "allhabad": "Allahabad",
                                 "alogorithms": "algorithms", "amenclinics": "amen clinics",
                                 "amendmending": "amend mending", "americanmedicalassoc": "american medical assoc",
                                 "amette": "annette", "ammusement": "amusement", "amorphus": "amorph us",
                                 "anarold": "Android", "andKyokushin": "and Kyokushin", "androneda": "andromeda",
                                 "animlaistic": "animalistic", "anmuslim": "an muslim", "annilingus": "anilingus",
                                 "annoincement": "announcement", "annonimous": "anonymous",
                                 "anomymously": "anonymously", "anonimously": "anonymously",
                                 "anti cipation": "anticipation", "anti-Semitic": "anti-semitic",
                                 "antibigots": "anti bigots",
                                 "antibrahmin": "anti brahmin", "anticancerous": "anti cancerous",
                                 "anticoncussive": "anti concussive", "antihindus": "anti hindus",
                                 "antilife": "anti life", "antimeter": "anti meter", "antireligion": "anti religion",
                                 "antivirius": "antivirus", "anxietymake": "anxiety make",
                                 "anyonestudied": "anyone studied",
                                 "aomeone": "someone", "apartheidisrael": "apartheid israel",
                                 "appsmoment": "apps moment", "aquous": "aqueous", "arceius": "Arcesius",
                                 "arceous": "araceous", "archaemenid": "Achaemenid", "arectifier": "rectifier",
                                 "aren't": "are not", "arencome": "aren come",
                                 "arewhatsapp": "are WhatsApp", "argruments": "arguments",
                                 "argumentetc": "argument etc", "aristocracylifestyle": "aristocracy lifestyle",
                                 "armanents": "armaments", "armenains": "armenians", "aroused21000": "aroused 21000",
                                 "arragent": "arrogant", "asexaul": "asexual", "assignmentcanyon": "assignment canyon",
                                 "aswell": "as well", "athiust": "athirst", "atlous": "atrous",
                                 "atonomous": "autonomous", "atrocitties": "atrocities", "attirements": "attire ments",
                                 "attrected": "attracted", "augumentation": "argumentation", "augumented": "augmented",
                                 "august2017": "august 2017",
                                 "aulphate": "sulphate", "austentic": "austenitic", "austinlizards": "austin lizards",
                                 "austroloid": "australoid", "austronauts": "astronauts", "autoliker": "auto liker",
                                 "autolikers": "auto likers", "autosexual": "auto sexual", "avacous": "vacuous",
                                 "avegeta": "ave geta",
                                 "awrong": "aw rong", "aysnchronous": "asynchronous", "aɴᴅ": "and", "aᴛ": "at",
                                 "b**": "bit", "b***": "bitc", "b****": "bitch", "b*ll": "bull", "b*tch": "bitch",
                                 "babymust": "baby must",
                                 "bacteries": "batteries", "badeffects": "bad effects",
                                 "badgermoles": "enormous, blind mammal", "badminaton": "badminton",
                                 "badmothing": "badmouthing", "badtameezdil": "badtameez dil",
                                 "balanoglossus": "Balanoglossus", "balckwemen": "balck women",
                                 "ballonets": "ballo nets", "banggood": "bang good",
                                 "bapus": "bapu s", "batmanvoice": "batman voice", "batsmencould": "batsmen could",
                                 "batteriesplus": "batteries plus", "beacsuse": "because", "becausr": "because",
                                 "becomesdouble": "becomes double", "becone": "become", "belagola": "bela gola",
                                 "beloney": "boloney",
                                 "beltholders": "belt holders", "bengolis": "Bengalis", "betterDtu": "better Dtu",
                                 "betterv3": "better", "betweenmanagement": "between management", "bhlushes": "blushes",
                                 "bi*ch": "bitch", "bigly": "big league", "bigolive": "big olive",
                                 "biromantical": "bi romantical",
                                 "bit*h": "bitch", "bitc*": "bitch", "bitcjes": "bitches", "bittergoat": "bitter goat",
                                 "blackbeauty": "black beauty", "blackboxing": "black boxing",
                                 "blackdotes": "black dotes", "blackmarks": "black marks", "blackmoney": "black money",
                                 "blackpaper": "black paper",
                                 "blackpink": "black pink", "blackpower": "black power",
                                 "blackwashing": "black washing", "blindfoldedly": "blindfolded",
                                 "blockchains": "blockchain", "blocktime": "block time", "bodhidharman": "Bodhidharma",
                                 "boldspot": "bolds pot", "bonesetters": "bone setters", "bonespur": "bones pur",
                                 "boothworld": "booth world", "bootyplus": "booty plus", "bootythongs": "booty thongs",
                                 "bramanistic": "Brahmanistic", "brightiest": "brightest", "brlieve": "believe",
                                 "brophytes": "bryophytes", "brotherzone": "brother zone",
                                 "brotherzoned": "brother zoned", "bugdget": "budget",
                                 "bushiri": "Bushire", "businessbor": "business bor",
                                 "businessinsider": "business insider", "businiss": "business", "bussiest": "fussiest",
                                 "bussinessmen": "businessmen", "bussinss": "bussings", "bustees": "bus tees",
                                 "c'mon": "come on", "caausing": "causing",
                                 "cakemaker": "cake maker", "calcalus": "calculus", "calccalculus": "calc calculus",
                                 "cammando": "commando", "camponente": "component", "campusthrough": "campus through",
                                 "campuswith": "campus with", "can't": "cannot", "canMuslims": "can Muslims",
                                 "canterlever": "canter lever",
                                 "capesindia": "capes india", "capuletwant": "capulet want",
                                 "careerplus": "career plus", "carnivorus": "carnivorous",
                                 "carnivourous": "carnivorous", "casterating": "castrating", "catagorey": "category",
                                 "categoried": "categories", "catrgory": "category", "catwgorized": "categorized",
                                 "cau.sing": "causing", "causinng": "causing", "ceftriazone": "ceftriaxone",
                                 "celcious": "delicious", "celemente": "Clemente", "celibatess": "celibates",
                                 "cellsius": "celsius", "celsious": "cesious", "cemenet": "cement",
                                 "centimiters": "centimeters",
                                 "cetusplay": "cetus play", "changetip": "change tip", "chaparone": "chaperone",
                                 "chapterwise": "chapter wise", "checkusers": "check users",
                                 "cheekboned": "cheek boned", "chestbusters": "chest busters", "cheverlet": "cheveret",
                                 "chigoe": "sub-tropical climates", "chinawares": "china wares",
                                 "chinesese": "chinese", "chiropractorone": "chiropractor one",
                                 "chlamydomanas": "chlamydomonas", "chromonema": "chromo nema", "chylus": "chylous",
                                 "circumradius": "circum radius", "ciswomen": "cis women",
                                 "citiesbetter": "cities better", "citruspay": "citrus pay", "clause55": "clause",
                                 "claustophobic": "claustrophobic", "clevercoyote": "clever coyote",
                                 "cloudways": "cloud ways", "clusture": "culture", "cobditioners": "conditioners",
                                 "cobustion": "combustion", "coclusion": "conclusion", "coincedences": "coincidences",
                                 "coincidents": "coincidence", "coinfirm": "confirm",
                                 "coinsidered": "considered", "coinsized": "coin sized", "coinstop": "coins top",
                                 "cointries": "countries", "cointry": "country", "colmbus": "columbus",
                                 "comandant": "commandant", "come2": "come to", "comepleted": "completed",
                                 "cometitive": "competitive",
                                 "comfortzone": "comfort zone", "comlementry": "complementary",
                                 "commencial": "commercial", "commensalisms": "commensal isms",
                                 "commissiioned": "commissioned", "commissionets": "commissioners",
                                 "commudus": "Commodus", "comodies": "corodies", "compartmentley": "compartment",
                                 "complusary": "compulsory",
                                 "componendo": "compon endo", "conciousnes": "conciousness",
                                 "conciousnesss": "consciousnesses", "conclusionless": "conclusion less",
                                 "concuous": "conscious", "condioner": "conditioner", "condtioners": "conditioners",
                                 "conecTU": "connect you", "conectiin": "connection", "conents": "contents",
                                 "confiment": "confident", "confousing": "confusing", "congusion": "confusion",
                                 "conpartment": "compartment", "consciousuness": "consciousness",
                                 "conscoiusness": "consciousness", "consicious": "conscious",
                                 "constiously": "consciously",
                                 "constitutionaldevelopment": "constitutional development",
                                 "contestious": "contentious",
                                 "contiguious": "contiguous", "contraproductive": "contra productive",
                                 "cooerate": "cooperate", "cooktime": "cook time", "cooldrink": "cool drink",
                                 "coreligionist": "co religionist", "corpgov": "corp gov", "cosicous": "conscious",
                                 "could've": "could have", "couldn't": "could not",
                                 "counciousness": "conciousness", "countinous": "continuous",
                                 "countryHow": "country How", "countryball": "country ball",
                                 "countryless": "Having no country", "countrypeople": "country people",
                                 "cousan": "cousin", "cousera": "couler", "cousi ": "cousin ", "covfefe": "coverage",
                                 "cracksbecause": "cracks because", "crazymaker": "crazy maker",
                                 "creditdation": "Accreditation", "cretinously": "cretinous",
                                 "cryptochristians": "crypto christians", "cryptoworld": "crypto world",
                                 "cuckholdry": "cuckold", "culturr": "culture", "cushelle": "cush elle",
                                 "customshoes": "customs hoes",
                                 "cutaneously": "cu taneously", "cyclohexenone": "cyclohexanone", "d***": "dick",
                                 "daigonal": "diagonal", "dailycaller": "daily caller", "danergous": "dangerous",
                                 "dangergrous": "dangerous", "dangorous": "dangerous", "darkweb": "dark web",
                                 "darkwebcrawler": "dark webcrawler",
                                 "darskin": "dark skin", "dealignment": "de alignment", "deangerous": "dangerous",
                                 "deartment": "department", "deathbecomes": "death becomes", "defpush": "def push",
                                 "degenarate": "degenerate", "degenerous": "de generous",
                                 "deindustralization": "deindustrialization", "dejavus": "dejavu s",
                                 "demanters": "decanters", "demantion": "detention", "demigirl": "demi girl",
                                 "demihuman": "demi human", "demisexuality": "demi sexuality",
                                 "demisexuals": "demisexual", "demonentization": "demonetization",
                                 "demonetised": "demonetized", "demonetistaion": "demonetization",
                                 "demonetizations": "demonetization",
                                 "demonetize": "demonetized", "demonetizedd": "demonetized",
                                 "demonetizing": "demonetized", "denomenator": "denominator",
                                 "departmentHow": "department How", "deplorables": "deplorable",
                                 "deployements": "deployments", "depolyment": "deployment",
                                 "dermatillomaniac": "dermatillomania", "desemated": "decimated",
                                 "designation-": "designation", "determenism": "determinism",
                                 "determenistic": "deterministic", "develepoments": "developments",
                                 "developmenent": "development", "developmeny": "development",
                                 "deverstion": "diversion", "devlelpment": "development", "devopment": "development",
                                 "di**": "dick",
                                 "dictioneries": "dictionaries", "didn't": "did not", "diffenky": "differently",
                                 "diffussion": "diffusion", "difigurement": "disfigurement", "difusse": "diffuse",
                                 "digitizeindia": "digitize india", "digonal": "di gonal", "digonals": "diagonals",
                                 "digustingly": "disgustingly",
                                 "diiscuss": "discuss", "dilusion": "delusion", "dimensionalise": "dimensional ise",
                                 "dimensiondimensions": "dimension dimensions", "dimensonless": "dimensionless",
                                 "dimensons": "dimensions", "dingtone": "ding tone", "dioestrus": "di oestrus",
                                 "diphosphorus": "di phosphorus", "diplococcus": "diplo coccus",
                                 "disagreemen": "disagreement", "disagreementt": "disagreement",
                                 "disagreementts": "disagreements", "disangagement": "disengagement",
                                 "disastrius": "disastrous", "disclousre": "disclosure",
                                 "discrimantion": "discrimination", "discriminatein": "discrimination",
                                 "disengenuously": "disingenuously", "dishonerable": "dishonorable",
                                 "disngush": "disgust", "disquss": "discuss", "disscused": "discussed",
                                 "disulphate": "di sulphate", "doccuments": "documents", "docoment": "document",
                                 "doctrne": "doctrine", "docuements": "documents", "docyments": "documents",
                                 "doesGauss": "does Gauss",
                                 "doesn't": "does not", "dogmans": "dogmas", "dogooder": "do gooder",
                                 "dolostones": "dolostone", "don't": "do not", "donaldtrumping": "donald trumping",
                                 "donesnt": "doesnt", "dood-": "dood", "doublelife": "double life",
                                 "dragonflag": "dragon flag",
                                 "dragonglass": "dragon glass", "dragonknight": "dragon knight",
                                 "dream11": " fantasy sports platform in India ", "drparment": "department",
                                 "dsymenorrhoea": "dysmenorrhoea", "dumbassess": "dumbass", "duramen": "dura men",
                                 "durgahs": "durgans", "dusyant": "distant", "e.g.": "for example",
                                 "eLitmus": "Indian company that helps companies in hiring employees",
                                 "earvphone": "earphone", "easedeverything": "eased everything",
                                 "eauipments": "equipments", "ecoworld": "eco world", "ecumencial": "ecumenical",
                                 "eduacated": "educated", "egomanias": "ego manias",
                                 "electronegativty": "electronegativity", "electroneum": "electro neum",
                                 "elementsbond": "elements bond", "eletronegativity": "electronegativity",
                                 "elinment": "eloinment", "embaded": "embased", "emellishments": "embellishments",
                                 "emiratis": "emirates", "emmenagogues": "emmenagogue",
                                 "employmentnews": "employment news", "encironment": "environment",
                                 "endrosment": "endorsement",
                                 "eneyone": "anyone", "engangement": "engagement", "enivironment": "environment",
                                 "enjoiment": "enjoyment", "enliightenment": "enlightenment",
                                 "enrollnment": "enrollment", "entaglement": "entanglement",
                                 "entartaiment": "entertainment", "enthusiasmless": "enthusiasm less",
                                 "entitlments": "entitlements",
                                 "entusiasta": "enthusiast", "envinment": "environment", "enviorement": "environment",
                                 "envioronments": "environments", "envirnmetalists": "environmentalists",
                                 "environmentai": "environmental", "envisionment": "envision ment",
                                 "ergophobia": "ergo phobia", "ergosphere": "ergo sphere",
                                 "ergotherapy": "ergo therapy",
                                 "ernomous": "enormous", "errounously": "erroneously", "et.al": "elsewhere",
                                 "ethethnicitesnicites": "ethnicity", "evangilitacal": "evangelical",
                                 "evenafter": "even after", "eventhogh": "even though", "everbodys": "everybody",
                                 "everperson": "ever person", "everydsy": "everyday",
                                 "everyfirst": "every first", "everyo0 ne": "everyone", "everyonr": "everyone",
                                 "exMuslims": "Ex-Muslims", "examsexams": "exams exams", "exause": "excuse",
                                 "exclusinary": "exclusionary", "excritment": "excitement",
                                 "exilarchate": "exilarch ate", "exmuslims": "ex muslims",
                                 "experimently": "experiment", "expertthink": "expert think", "expessway": "expressway",
                                 "exserviceman": "ex serviceman", "exservicemen": "ex servicemen",
                                 "extemporeneous": "extemporaneous", "extraterestrial": "extraterrestrial",
                                 "extroneous": "extraneous", "exust": "ex ust", "eyemake": "eye make",
                                 "f**": "fuc", "f***": "fuck", "f**k": "fuck", "fAegon": "wagon",
                                 "faidabad": "Faizabad", "falaxious": "fallacious", "fallicious": "fallacious",
                                 "famousbirthdays": "famous birthdays", "fanessay": "fan essay",
                                 "fanmenow": "fan menow",
                                 "farmention": "farm ention", "fecetious": "facetious", "feelinfs": "feelings",
                                 "feelingstupid": "feeling stupid", "feelomgs": "feelings", "feelonely": "feel onely",
                                 "feelwhen": "feel when", "femanists": "feminists", "femenise": "feminise",
                                 "femenism": "feminism",
                                 "femenists": "feminists", "feminisam": "feminism", "feminists": "feminism supporters",
                                 "fentayal": "fentanyl", "fergussion": "percussion", "fermentqtion": "fermentation",
                                 "fevelopment": "development", "fewshowanyRemorse": "few show any Remorse",
                                 "filabustering": "filibustering", "findamental": "fundamental",
                                 "fingols": "Finnish people are supposedly descended from Mongols",
                                 "fisgeting": "fidgeting", "fislike": "dislike", "fitgirl": "fit girl",
                                 "flashgive": "flash give", "flemmings": "flemming",
                                 "followingorder": "following order", "fondels": "fondles", "fonecare": "f onecare",
                                 "foodlovee": "food lovee",
                                 "forcestop": "forces top", "forgetfulnes": "forgetfulness",
                                 "forgottenfaster": "forgotten faster", "fraysexual": "fray sexual",
                                 "friedzone": "fried zone", "frighter": "fighter", "fromGermany": "from Germany",
                                 "fronend": "friend", "frozen tamod": "Pornographic website", "frustratd": "frustrate",
                                 "fu.k": "fuck", "fuckboys": "fuckboy", "fuckgirl": "fuck girl",
                                 "fuckgirls": "fuck girls", "fucktrumpet": "fuck trumpet", "fudamental": "fundamental",
                                 "fundemantally": "fundamentally", "fundsindia": "funds india", "gadagets": "gadgets",
                                 "gadgetsnow": "gadgets now",
                                 "gamergaye": "gamersgate", "gamersexual": "gamer sexual",
                                 "gangstalkers": "gangs talkers", "garycrum": "gary crum", "gaushalas": "gaus halas",
                                 "gayifying": "gayed up with homosexual love", "gayke": "gay Online retailers",
                                 "geerymandered": "gerrymandered", "geminatus": "geminates", "genetilia": "genitalia",
                                 "geniusses": "geniuses", "genocidizing": "genociding", "genuiuses": "geniuses",
                                 "geomentrical": "geometrical", "geramans": "germans", "germanized": "become german",
                                 "germanophone": "Germanophobe", "germeny": "Germany", "get1000": "get 1000",
                                 "get630": "get 630",
                                 "get90": "get", "getDepersonalization": "get Depersonalization",
                                 "getallparts": "get allparts", "getdrip": "get drip", "getfinancing": "get financing",
                                 "getile": "gentile", "getmy": "get my", "getsmuggled": "get smuggled",
                                 "gettibg": "getting", "gettubg": "getting",
                                 "gevernment": "government", "ghazibad": "ghazi bad", "girlfrnd": "girlfriend",
                                 "girlriend": "girlfriend", "glocuse": "louse", "glrous": "glorious",
                                 "glueteus": "gluteus", "goalwise": "goal wise", "goatnuts": "goat nuts",
                                 "goegraphies": "geographies",
                                 "gogoro": "gogo ro", "golddig": "gold dig", "goldengroup": "golden group",
                                 "goldmedal": "gold medal", "goldquest": "gold quest", "golemized": "polemized",
                                 "golusu": "gol usu", "gomovies": "go movies", "gonverment": "government",
                                 "goodfirms": "good firms",
                                 "goodfriends": "good friends", "goodspeaks": "good speaks", "google4": "google",
                                 "googlemapsAPI": "googlemaps API", "googology": "online encyclopedia",
                                 "googolplexain": "googolplexian", "gorakpur": "Gorakhpur", "gorgops": "gorgons",
                                 "gorichen": "Gori Chen Mountain", "gorlfriend": "girlfriend",
                                 "got7": "got", "gothras": "goth ras", "gouing": "going", "goundar": "Gondar",
                                 "gouverments": "governments", "goverenments": "governments",
                                 "govermenent": "government", "governening": "governing", "governmaent": "government",
                                 "govnments": "movements",
                                 "govrment": "government", "govshutdown": "gov shutdown",
                                 "govtribe": "provides real-time federal contracting market intel", "gpusli": "gusli",
                                 "greatuncle": "great uncle", "greenhouseeffect": "greenhouse effect",
                                 "greenseer": "people who possess the magical ability", "greysexual": "grey sexual",
                                 "gridgirl": "female models of the race", "gujaratis": "gujarati",
                                 "gujratis": "gujarati", "h***": "hole", "h*ck": "hack",
                                 "habius - corpus": "habeas corpus", "hadn't": "had not",
                                 "halfgirlfriend": "half girlfriend", "handgloves": "hand gloves",
                                 "haoneymoon": "honeymoon", "haraasment": "harassment", "harashment": "harassment",
                                 "harasing": "harassing", "harassd": "harassed", "harassument": "harassment",
                                 "harbaceous": "herbaceous", "hasn't": "has not", "hasserment": "Harassment",
                                 "hatrednesss": "hatred", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                                 "he's": "he is", "heeadphones": "headphones", "heightism": "height discrimination",
                                 "heineous": "heinous", "heitus": "Leitus", "helisexual": "sexual", "here's": "here is",
                                 "heriones": "heroines", "hetereosexual": "heterosexual",
                                 "heteromantic": "hete romantic",
                                 "heteroromantic": "hetero romantic", "hetorosexuality": "hetoro sexuality",
                                 "hetorozygous": "heterozygous", "hetrogenous": "heterogenous", "hidusim": "hinduism",
                                 "hillum": "helium", "himanity": "humanity", "himdus": "hindus", "hindhus": "hindus",
                                 "hindian": "North Indian",
                                 "hindians": "North Indian", "hindusm": "hinduism", "hippocratical": "Hippocratical",
                                 "hkust": "hust", "hollcaust": "holocaust", "holocause": "holocaust",
                                 "holocaustable": "holocaust", "holocaustal": "holocaust",
                                 "holocausting": "holocaust ing", "homageneous": "homogeneous",
                                 "homeseek": "home seek", "homogeneus": "homogeneous", "homologus": "homologous",
                                 "homoromantic": "homo romantic", "homosexualtiy": "homosexuality",
                                 "homosexulity": "homosexuality", "homosporous": "homos porous",
                                 "honorkillings": "honor killings", "hotelmanagement": "hotel management",
                                 "housban": "Housman",
                                 "householdware": "household ware", "housepoor": "house poor", "how'd": "how did",
                                 "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "howhat": "how that",
                                 "htderabad": "Hyderabad", "htmlutmterm": "html utm term", "huemanity": "humanity",
                                 "hukous": "humous", "humancoalition": "human coalition", "humanfemale": "human female",
                                 "humanitariarism": "humanitarianism", "humanzees": "human zees",
                                 "humilates": "humiliates", "hurrasement": "hurra sement", "husoone": "huso one",
                                 "hyperfocusing": "hyper focusing", "hypersexualize": "hyper sexualize",
                                 "hyperthalamus": "hyper thalamus", "hypogonadic": "hypogonadia",
                                 "hypotenous": "hypogenous", "hystericus": "hysteric us", "i'd": "i would",
                                 "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                                 "i've": "i have",
                                 "i.e.": "in other words", "iPhone6": "iPhone", "iPhone7": "iPhone 7",
                                 "iPhoneX": "iPhone", "ibdustries": "industries", "icrement": "increment",
                                 "idoot": "idiot", "ignoramouses": "ignoramuses", "illigitmate": "illegitimate",
                                 "illiustrations": "illustrations",
                                 "illlustrator": "illustrator", "illustratuons": "illustrations",
                                 "ilustrating": "illustrating", "immplemented": "implemented", "improment": "impriment",
                                 "imrovment": "improvement", "imrpovement": "improvement",
                                 "inapropriately": "inappropriately", "incestious": "incestuous",
                                 "increaments": "increments",
                                 "indans": "indians", "indemendent": "independent", "indemenity": "indemnity",
                                 "indianarmy": "indian army", "indianleaders": "indian leaders",
                                 "indiareads": "india reads", "indiarush": "india rush", "indiginious": "indigenous",
                                 "indominus": "in dominus", "indrold": "Android",
                                 "indrustry": "industry", "indtrument": "instrument", "industdial": "industrial",
                                 "industey": "industry", "industion": "induction", "industiry": "industry",
                                 "industrZgies": "industries", "industrilaization": "industrialization",
                                 "industrilised": "industrialised", "industrilization": "industrialization",
                                 "industris": "industries", "industrybuying": "industry buying",
                                 "infrctuous": "infectuous", "infridgement": "infringement",
                                 "infringiment": "infringement", "infrustructre": "infrastructure",
                                 "infussions": "infusions", "infustry": "industry", "inheritantly": "inherently",
                                 "innermongolians": "inner mongolians",
                                 "innerskin": "inner skin", "inpersonations": "impersonations",
                                 "inplementing": "implementing", "inpregnated": "in pregnant", "inradius": "in radius",
                                 "insectivourous": "insectivorous", "insemenated": "inseminated",
                                 "insipudus": "insipidus", "instagate": "instigate", "instantanous": "instantaneous",
                                 "instantneous": "instantaneous", "instatanously": "instantaneously",
                                 "instrumenot": "instrument", "instrumentalizing": "instrument",
                                 "instrumention": "instrument ion", "insustry": "industry", "integumen": "integument",
                                 "intelliegent": "intelligent", "intermate": "inter mating",
                                 "interpersonation": "inter personation",
                                 "intrasegmental": "intra segmental", "intrusivethoughts": "intrusive thoughts",
                                 "intwrsex": "intersex", "inudustry": "industry", "inuslating": "insulating",
                                 "invement": "movement", "inverstment": "investment",
                                 "investmentguru": "investment guru", "invetsment": "investment",
                                 "involvedinthe": "involved in the",
                                 "invovement": "involvement", "iodoketones": "iodo ketones",
                                 "iodopovidone": "iodo povidone", "iphone6": "iPhone", "iphone7": "iPhone",
                                 "iphone8": "iPhone", "iphoneX": "iPhone", "irakis": "iraki",
                                 "irreputable": "reputation", "irrevershible": "irreversible",
                                 "isUnforgettable": "is Unforgettable", "iservant": "servant",
                                 "islamphobia": "islam phobia", "islamphobic": "islam phobic", "isn't": "is not",
                                 "isotones": "iso tones", "ispakistan": "is pakistan", "isthmuses": "isthmus es",
                                 "it'd": "it would", "it'd've": "it would have",
                                 "it'll": "it will", "it'll've": "it will have", "it's": "it is", "itstead": "instead",
                                 "iusso": "kusso", "javadiscussion": "java discussion", "jeaslous": "jealous",
                                 "jeesyllabus": "jee syllabus", "jerussalem": "jerusalem", "jetbingo": "jet bingo",
                                 "jewprofits": "jew profits", "jewsplain": "jews plain", "jionee": "jinnee",
                                 "joboutlook": "job outlook", "journalust": "journalist", "judisciously": "judiciously",
                                 "jusify": "justify", "jusrlt": "just", "justcheaptickets": "just cheaptickets",
                                 "justcreated": "just created",
                                 "justdile": "justice", "justiciaries": "justiciary", "justyfied": "justified",
                                 "justyfy": "justify", "kabadii": "kabaddi", "katgmandu": "katmandu",
                                 "keralapeoples": "kerala peoples", "kerataconus": "keratoconus",
                                 "keratokonus": "keratoconus", "keywordseverywhere": "keywords everywhere",
                                 "killedshivaji": "killed shivaji", "killikng": "killing", "killograms": "kilograms",
                                 "killyou": "kill you", "knowble": "Knowle", "knowldage": "knowledge",
                                 "knowledeg": "knowledge", "knowledgd": "knowledge",
                                 "knowledgeWoods": "knowledge Woods", "knowlefge": "knowledge",
                                 "knowlegdeable": "knowledgeable", "knownprogramming": "known programming",
                                 "knowyouve": "know youve", "kotatsus": "kotatsu s", "kushanas": "kusha nas",
                                 "ladymen": "ladyboy", "lamenectomy": "lamnectomy",
                                 "laowhy86": "Foreigners who do not respect China", "lawforcement": "law forcement",
                                 "lawyergirlfriend": "lawyer girl friend",
                                 "legumnous": "leguminous", "let's": "let us", "lethocerus": "Lethocerus",
                                 "lexigographers": "lexicographers", "lifeaffect": "life affect",
                                 "liferature": "literature", "lifestly": "lifestyle", "lifestylye": "lifestyle",
                                 "lifeute": "life ute", "likebJaish": "like bJaish",
                                 "likelogo": "like logo", "likelovequotes": "like lovequotes", "likemail": "like mail",
                                 "likemy": "like my", "like⬇": "like", "lionese": "lioness",
                                 "lithromantics": "lith romantics", "lndustrial": "industrial",
                                 "loadingtimes": "loading times", "lonewolfs": "lone wolfs",
                                 "loundly": "loudly", "louspeaker": "loudspeaker", "lovejihad": "love jihad",
                                 "lovestep": "love step", "lucideus": "lucidum", "ludiculous": "ridiculous",
                                 "lustfly": "lustful", "lustorus": "lustrous", "ma'am": "madam",
                                 "macapugay": "Macaulay",
                                 "machineworld": "machine world", "magibabble": "magi babble",
                                 "mailwoman": "mail woman", "mainstreamly": "mainstream", "makedonian": "macedonian",
                                 "makeschool": "make school", "makeshifter": "make shifter", "makeup411": "makeup 411",
                                 "mamgement": "management", "manaagerial": "managerial",
                                 "managemebt": "management", "managemenet": "management",
                                 "managemental": "manage mental", "managementskills": "management skills",
                                 "managersworking": "managers working", "managewp": "managed",
                                 "manajement": "management", "manaufacturing": "manufacturing",
                                 "mandalikalu": "mandalika lu", "mandateing": "man dateing",
                                 "mandatkry": "mandatory", "mandingan": "Mandingan", "mandrillaris": "mandrill aris",
                                 "maneuever": "maneuver", "mangalasutra": "mangalsutra", "mangalik": "manglik",
                                 "mangekyu": "mange kyu", "mangolian": "mongolian", "mangoliod": "mongoloid",
                                 "mangonada": "mango nada",
                                 "manholding": "man holding", "manhuas": "mahuas", "manies": "many",
                                 "manipative": "mancipative", "manipulant": "manipulate", "manipullate": "manipulate",
                                 "maniquins": "mani quins", "manjha": "mania", "mankirt": "mankind",
                                 "mankrit": "mank rit",
                                 "manlet": "man let", "manniya": "mania", "mannualy": "annual",
                                 "manorialism": "manorial ism", "manpads": "man pads", "manrega": "Manresa",
                                 "mansatory": "mandatory", "manslamming": "mans lamming", "mansoon": "man soon",
                                 "manspread": "man spread",
                                 "manspreading": "man spreading", "manstruate": "menstruate",
                                 "mansturbate": "masturbate", "manterrupting": "interrupting", "manthras": "mantras",
                                 "manufacctured": "manufactured", "manufacturig": "manufacturing",
                                 "manufctures": "manufactures", "manufraturer": "manufacturer",
                                 "manufraturing": "manufacturing",
                                 "manufucturing": "manufacturing", "manupalation": "manipulation",
                                 "manupulative": "manipulative", "manvantar": "Manvantara", "manwould": "man would",
                                 "manwues": "manages", "many4": "many", "manyare": "many are", "manychat": "many chat",
                                 "manycountries": "many countries",
                                 "manygovernment": "many government", "manyness": "many ness",
                                 "manyother": "many other", "maralago": "Mar-a-Lago", "maratis": "Maratism",
                                 "marionettist": "Marionettes", "marlstone": "marls tone", "masskiller": "mass killer",
                                 "mastubrate": "masturbate", "mastuburate": "masturbate",
                                 "mauritious": "mauritius", "mausturbate": "masturbate", "mayn't": "may not",
                                 "mcgreggor": "McGregor", "measument": "measurement", "meausrements": "measurements",
                                 "medicalperson": "medical person", "meesaya": "mee saya", "megabuses": "mega buses",
                                 "mellophones": "mellophone s",
                                 "memoney": "money", "menberships": "memberships", "mendalin": "mend alin",
                                 "mendatory": "mandatory", "menditory": "mandatory", "menedatory": "mandatory",
                                 "menifest": "manifest", "menifesto": "manifesto", "menigioma": "meningioma",
                                 "meninist": "male chauvinism",
                                 "meniss": "menise", "menmium": "medium", "mensrooms": "mens rooms",
                                 "menstrat": "menstruate", "menstrated": "menstruated", "menstraution": "menstruation",
                                 "menstruateion": "menstruation", "menstrute": "menstruate",
                                 "menstrution": "menstruation", "menstruual": "menstrual",
                                 "menstuating": "menstruating", "mensurational": "mensuration al",
                                 "mentalitiy": "mentality", "mentalized": "metalized", "mentenance": "maintenance",
                                 "mentionong": "mentioning", "mentiri": "entire", "meritious": "meritorious",
                                 "merrigo": "merligo", "messenget": "messenger",
                                 "metacompartment": "meta compartment", "metaphosphates": "meta phosphates",
                                 "methedone": "methadone", "michrophone": "microphone",
                                 "microaggression": "micro aggression", "microapologize": "micro apologize",
                                 "microneedling": "micro needling", "microservices": "micro services",
                                 "microskills": "micros kills", "middleperson": "middle person",
                                 "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                                 "milkwithout": "milk without", "minderheid": "minder worse", "misandrous": "misandry",
                                 "mismanagements": "mis managements", "momsays": "moms ays", "momuments": "monuments",
                                 "monetation": "moderation",
                                 "moneycard": "money card", "moneycash": "money cash", "moneydriven": "money driven",
                                 "moneyof": "mony of", "mongodump": "mongo dump", "mongorestore": "mongo restore",
                                 "monoceious": "monoecious", "monogmous": "monogamous",
                                 "monosexuality": "mono sexuality", "montheistic": "nontheistic",
                                 "moraled": "morale", "motorious": "notorious", "mountaneous": "mountainous",
                                 "mousestats": "mouses tats", "mousquitoes": "mosquitoes",
                                 "moustachesomething": "moustache something", "moustachess": "moustaches",
                                 "movielush": "movie lush", "moviments": "movements", "msitake": "mistake",
                                 "muccus": "mucous", "muchdeveloped": "much developed", "muscleblaze": "muscle blaze",
                                 "muscluar": "muscular", "muscualr": "muscular", "musigma": "mu sigma",
                                 "musilim": "Muslim", "musilms": "muslims", "musims": "Muslims",
                                 "muslimare": "Muslim are",
                                 "muslisms": "muslims", "muslium": "muslim", "mussraff": "muss raff",
                                 "must've": "must have", "mustabating": "must abating", "mustang1": "mustangs",
                                 "mustectomy": "mastectomy", "mustn't": "must not", "mustn't've": "must not have",
                                 "musturbation": "masturbation",
                                 "musuclar": "muscular", "mutiliating": "mutilating", "muzaffarbad": "muzaffarabad",
                                 "mylogenous": "myogenous", "mystakenly": "mistakenly", "mythomaniac": "mythomania",
                                 "nagetive": "native", "naggots": "faggots", "namaj": "namaz",
                                 "narcsissist": "narcissist",
                                 "narracist": "nar racist", "narsistic": "narcistic", "nationalpost": "national post",
                                 "nauget": "naught", "nauseatic": "nausea tic", "neculeus": "nucleus",
                                 "needn't": "need not", "needn't've": "need not have", "negetively": "negatively",
                                 "negosiation": "negotiation",
                                 "negotiatiations": "negotiations", "negotiatior": "negotiation",
                                 "negotiotions": "negotiations", "neigous": "nervous", "nesraway": "nearaway",
                                 "nestaway": "nest away", "neucleus": "nucleus", "neurosexist": "neuro sexist",
                                 "neuseous": "nauseous", "neverhteless": "nevertheless",
                                 "neverunlearned": "never unlearned", "newkiller": "new killer",
                                 "newscommando": "news commando", "nicholus": "nicholas", "nicus": "nidus",
                                 "nitrogenious": "nitrogenous", "nomanclature": "nomenclature",
                                 "nonHindus": "non Hindus", "nonabandonment": "non abandonment",
                                 "noneconomically": "non economically",
                                 "nonelectrolyte": "non electrolyte", "nonexsistence": "nonexistence",
                                 "nonfermented": "non fermented", "nonskilled": "non skilled",
                                 "nonspontaneous": "non spontaneous", "nonviscuous": "nonviscous",
                                 "northcap": "north cap", "northestern": "northwestern", "nosebone": "nose bone",
                                 "nothinking": "thinking",
                                 "notmusing": "not musing", "nuchakus": "nunchakus", "nuclus": "nucleus",
                                 "nullifed": "nullified", "nuslims": "Muslims", "nutriteous": "nutritious",
                                 "o'clock": "of the clock", "obfuscaton": "obfuscation", "objectmake": "object make",
                                 "obnxious": "obnoxious",
                                 "ofKulbhushan": "of Kulbhushan", "ofVodafone": "of Vodafone",
                                 "ointmentsointments": "ointments ointments", "omnisexuality": "omni sexuality",
                                 "one300": "one 300", "oneblade": "one blade", "onecoin": "one coin",
                                 "onepiecedeals": "onepiece deals", "onsocial": "on social", "oogonial": "oogonia l",
                                 "orangetheory": "orange theory", "otherstates": "others tates",
                                 "oughtn't": "ought not", "oughtn't've": "ought not have", "ousmania": "ous mania",
                                 "outfocus": "out focus", "outonomous": "autonomous", "overcold": "over cold",
                                 "overcomeanxieties": "overcome anxieties", "overfeel": "over feel",
                                 "overjustification": "over justification", "overproud": "over proud",
                                 "overvcome": "overcome", "oxandrolonesteroid": "oxandrolone steroid",
                                 "ozonedepletion": "ozone depletion", "p***": "porn", "p****": "pussy", "p*rn": "porn",
                                 "p*ssy": "pussy", "p0 rnstars": "pornstars",
                                 "padmanabhanagar": "padmanabhan agar", "painterman": "painter man",
                                 "pakistanisbeautiful": "pakistanis beautiful", "palsmodium": "plasmodium",
                                 "palusami": "palus ami", "pangolinminer": "pangolin miner",
                                 "panishments": "punishments", "panromantic": "pan romantic", "pansexuals": "pansexual",
                                 "papermoney": "paper money",
                                 "paracommando": "para commando", "parasuramans": "parasuram ans",
                                 "parilment": "parchment", "parlamentarians": "parliamentarians",
                                 "parlamentary": "parliamentary", "parlementarian": "parlement arian",
                                 "parlimentry": "parliamentary", "parmenent": "permanent", "parmently": "patiently",
                                 "parralels": "parallels",
                                 "patitioned": "petitioned", "peactime": "peacetime", "pegusus": "pegasus",
                                 "peloponesian": "peloponnesian", "pentanone": "penta none",
                                 "peoplekind": "people kind", "peoplelike": "people like", "perfoemance": "performance",
                                 "performancelearning": "performance learning", "performancies": "performances",
                                 "permanentjobs": "permanent jobs", "permanmently": "permanently",
                                 "permenganate": "permanganate", "persoenlich": "person lich",
                                 "personaltiles": "personal titles", "personifaction": "personification",
                                 "personlich": "person lich", "personslized": "personalized",
                                 "persulphates": "per sulphates", "petrostates": "petro states",
                                 "pettypotus": "petty potus", "pharisaistic": "pharisaism", "phenonenon": "phenomenon",
                                 "pheramones": "pheromones", "philanderous": "philander ous",
                                 "phlegmonous": "phlegmon ous", "phnomenon": "phenomenon", "phonecases": "phone cases",
                                 "photoacoustics": "photo acoustics", "phusicist": "physicist",
                                 "phythagoras": "pythagoras", "picosulphate": "pico sulphate", "pingo5": "pingo",
                                 "pizza gate": "debunked conspiracy theory", "placdment": "placement",
                                 "plaement": "placement", "plagetum": "plage tum", "platfrom": "platform",
                                 "pleasegive": "please give", "plus5": "plus",
                                 "plustwo": "plus two", "poisenious": "poisonous", "poisiones": "poisons",
                                 "pokestops": "pokes tops", "polishment": "pol ishment", "politicak": "political",
                                 "polonious": "polonius", "polyagamous": "polygamous", "polygomists": "polygamists",
                                 "polygony": "poly gony",
                                 "polyhouse": "Polytunnel", "polyhouses": "Polytunnel", "poneglyphs": "pone glyphs",
                                 "posinous": "rosinous", "posionus": "poisons", "postincrement": "post increment",
                                 "postmanare": "postman are", "powerballsusa": "powerballs usa",
                                 "prechinese": "pre chinese", "prefomance": "performance",
                                 "pregnantwomen": "pregnant women", "preincrement": "pre increment",
                                 "prejusticed": "prejudiced", "prelife": "pre life", "prelimenary": "preliminary",
                                 "prendisone": "prednisone", "prentious": "pretentious",
                                 "presumptuousnes": "presumptuousness", "pretenious": "pretentious",
                                 "pretex": "pretext",
                                 "pretextt": "pre text", "preussure": "pressure", "previius": "previous",
                                 "primarty": "primary", "probationees": "probationers", "procument": "procumbent",
                                 "prodimently": "prominently", "productManagement": "product Management",
                                 "productionsupport": "production support", "productsexamples": "products examples",
                                 "programmebecause": "programme because",
                                 "programmingassignments": "programming assignments", "promuslim": "pro muslim",
                                 "propelment": "propel ment", "prospeorus": "prosperous",
                                 "prosporously": "prosperously", "protopeterous": "protopterous",
                                 "provocates": "provokes", "pseudomeningocele": "pseudo meningocele",
                                 "psiphone": "psi phone",
                                 "publious": "Publius", "pulmanery": "pulmonary", "puniahment": "punishment",
                                 "purusharthas": "purushartha", "purushottampur": "purushottam pur",
                                 "puspak": "pu spak", "pussboy": "puss boy", "pyromantic": "pyro mantic",
                                 "qualifeid": "qualified", "queffing": "queefing",
                                 "qurush": "qu rush", "r - apist": "rapist", "racistcomments": "racist comments",
                                 "racistly": "racist", "ramanunjan": "Ramanujan", "rammandir": "ram mandir",
                                 "rammayana": "ramayana", "rangeman": "range man", "rapefilms": "rape films",
                                 "rapiest": "rapist",
                                 "ratkill": "rat kill", "realbonus": "real bonus", "reallySemites": "really Semites",
                                 "realtimepolitics": "realtime politics", "recmommend": "recommend",
                                 "recommendor": "recommender", "recommening": "recommending",
                                 "recordermans": "recorder mans", "recrecommend": "rec recommend",
                                 "recruitment2017": "recruitment 2017",
                                 "recrument": "recrement", "recuirement": "requirement", "recurtment": "recurrent",
                                 "recusion": "recushion", "recussed": "recursed", "redicules": "ridiculous",
                                 "redius": "radius", "refundment": "refund ment", "regements": "regiments",
                                 "regilious": "religious",
                                 "reglamented": "reg lamented", "regognitions": "recognitions",
                                 "regognized": "recognized", "reimbrusement": "reimbursement", "rejectes": "rejected",
                                 "reliegious": "religious", "religiouslike": "religious like",
                                 "remaninder": "remainder", "remenant": "remnant", "remmaning": "remaining",
                                 "rendementry": "rendement ry", "repariments": "departments",
                                 "replecement": "replacement", "repugicans": "republicans",
                                 "repurcussion": "repercussion", "repyament": "repayment", "requairment": "requirement",
                                 "requriment": "requirement", "requriments": "requirements", "requsite": "requisite",
                                 "requsitioned": "requisitioned", "rescouses": "responses", "resigment": "resignment",
                                 "reskilled": "skilled", "resustence": "resistence", "retairment": "retainment",
                                 "reteirement": "retirement", "retrovisus": "retrovirus", "returement": "retirement",
                                 "reusibility": "reusability",
                                 "reusult": "result", "reverificaton": "reverification", "richfeel": "rich feel",
                                 "richmencupid": "rich men dating website", "ridicjlously": "ridiculously",
                                 "righteouness": "righteousness", "rigrously": "rigorously", "rivigo": "rivi go",
                                 "roadgods": "road gods", "rohmbus": "rhombus",
                                 "romantize": "romanize", "rombous": "bombous", "routez": "route",
                                 "roxycodone": "r oxycodone", "royago": "royal", "rseearch": "research",
                                 "rstman": "Rotman", "rubustness": "robustness", "russiagate": "russia gate",
                                 "russophobic": "Russophobiac",
                                 "s**": "shi", "s***": "shit", "saadus": "status",
                                 "saffronize": "India, politics, derogatory",
                                 "saffronized": "India, politics, derogatory", "sagittarious": "sagittarius",
                                 "salemmango": "salem mango", "salesmanago": "salesman ago", "sallatious": "fallacious",
                                 "samousa": "samosa",
                                 "sapiosexual": "Sexually attracted to intelligence", "sapiosexuals": "sapiosexual",
                                 "savegely": "savagely", "savethechildren": "save thechildren", "savonious": "sanious",
                                 "saydaw": "say daw", "saynthesize": "synthesize", "sayying": "saying",
                                 "scomplishments": "accomplishments", "scuduse": "scud use",
                                 "searious": "serious", "securedlife": "secured life", "secutus": "sects",
                                 "seeies": "series", "seekingmillionaire": "seeking millionaire", "see‬": "see",
                                 "selfknowledge": "self knowledge", "selfpayment": "self payment",
                                 "semisexual": "semi sexual", "serieusly": "seriously",
                                 "seriousity": "seriosity", "settelemen": "settlement",
                                 "settlementtake": "settlement take", "sevenpointed": "seven pointed",
                                 "seventysomething": "seventy something", "severiity": "severity",
                                 "seviceman": "serviceman", "sexeverytime": "sex everytime", "sexgods": "sex gods",
                                 "sexitest": "sexiest",
                                 "sexlike": "sex like", "sexmates": "sex mates", "sexond": "second",
                                 "sexpat": "sex tourism", "sexsurrogates": "sex surrogates", "sextactic": "sex tactic",
                                 "sexualSlutty": "sexual Slutty", "sexualises": "sexualise",
                                 "sexualityism": "sexuality ism", "sexuallly": "sexually",
                                 "sexuly": "sexily", "sexxual": "sexual", "sexyjobs": "sexy jobs", "sh*tty": "shit",
                                 "sha'n't": "shall not", "shan't": "shall not", "shan't've": "shall not have",
                                 "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                                 "she'll've": "she will have", "she's": "she is", "shitpeople": "shit people",
                                 "should've": "should have", "shouldn't": "should not",
                                 "shouldn't've": "should not have", "shouldntake": "shouldnt take", "shrusti": "shruti",
                                 "shuoldnt": "shouldnt", "silghtest": "slightest",
                                 "siliconindia": "silicon india", "silumtaneously": "simultaneously",
                                 "simantaneously": "simultaneously", "simentaneously": "simultaneously",
                                 "simoultaneously": "simultaneously", "simultameously": "simultaneously",
                                 "simultenously": "simultaneously", "simultqneously": "simultaneously",
                                 "simutaneusly": "simultaneously", "simutanously": "simultaneously",
                                 "sinophone": "Sinophobe", "sissyphus": "sisyphus", "sisterzoned": "sister zoned",
                                 "skillclasses": "skill classes", "skillselect": "skills elect", "skyhold": "sky hold",
                                 "slavetrade": "slave trade", "sllaybus": "syllabus", "sllyabus": "syllabus",
                                 "sllybus": "syllabus",
                                 "slogun": "slogan", "sloppers": "slippers", "slovenised": "slovenia",
                                 "smarthpone": "smartphone", "smartworld": "sm artworld", "smelllike": "smell like",
                                 "snapragon": "snapdragon", "snazzyway": "snazzy way", "sneakerlike": "sneaker like",
                                 "snuus": "snugs",
                                 "so's": "so as", "so've": "so have", "sociomantic": "sciomantic",
                                 "softskill": "softs kill", "soldiders": "soldiers", "someonewith": "some onewith",
                                 "southAfricans": "south Africans", "southeners": "southerners",
                                 "southerntelescope": "southern telescope", "spacewithout": "space without",
                                 "spause": "spouse", "speakingly": "speaking", "specrum": "spectrum",
                                 "spectulated": "speculated", "sponataneously": "spontaneously",
                                 "sponteneously": "spontaneously", "sportsusa": "sports usa", "sppliment": "supplement",
                                 "spymania": "spy mania", "sqamous": "squamous",
                                 "sreeman": "freeman", "st*up*id": "stupid", "staionery": "stationery",
                                 "stargold": "a Hindi movie channel", "starseeders": "star seeders",
                                 "stategovt": "state govt", "statemment": "statement", "statusGuru": "status Guru",
                                 "stillshots": "stills hots", "stillsuits": "still suits",
                                 "stomuch": "stomach", "stonepelters": "stone pelters", "stopings": "stoping",
                                 "stoppef": "stopped", "stoppingexercises": "stopping exercises",
                                 "stopsigns": "stop signs", "stopsits": "stop sits", "straitstimes": "straits times",
                                 "stupidedt": "stupidest", "stusy": "study",
                                 "subcautaneous": "subcutaneous", "subcentimeter": "sub centimeter",
                                 "subconcussive": "sub concussive", "subconsciousnesses": "sub consciousnesses",
                                 "subligamentous": "sub ligamentous", "suchvstupid": "such stupid",
                                 "suconciously": "unconciously", "sulmann": "Suilmann", "sulprus": "surplus",
                                 "sunnyleone": "sunny leone",
                                 "sunstop": "sun stop", "supermaneuverability": "super maneuverability",
                                 "supermaneuverable": "super maneuverable", "superplus": "super plus",
                                 "supersynchronous": "super synchronous", "supertournaments": "super tournaments",
                                 "supplemantary": "supplementary", "supplemenary": "supplementary",
                                 "supplementplatform": "supplement platform", "supplymentary": "supply mentary",
                                 "supplymentry": "supplementary", "surgetank": "surge tank",
                                 "susbtraction": "substraction", "suspectance": "suspect ance", "suspeect": "suspect",
                                 "suspenive": "suspensive", "suspicius": "suspicious", "sussessful": "successful",
                                 "sustainabke": "sustainable", "sustinet": "sustinent",
                                 "susubsoil": "su subsoil", "swayable": "sway able", "syallbus": "syllabus",
                                 "syallubus": "syllabus", "sychronous": "synchronous", "sylaabus": "syllabus",
                                 "sylabbus": "syllabus", "syllaybus": "syllabus", "tRump": "trump",
                                 "takeove": "takeover",
                                 "takeoverr": "takeover", "takeoverrs": "takeovers", "takesuch": "take such",
                                 "takingoff": "taking off", "talecome": "tale come", "tamanaa": "Tamanac", "taskus": "",
                                 "teamtreehouse": "team treehouse", "techinacal": "technical",
                                 "techmakers": "tech makers",
                                 "teethbrush": "teeth brush", "telegraphindia": "telegraph india",
                                 "terimanals": "terminals", "testostersone": "testosterone",
                                 "teststerone": "testosterone", "tetramerous": "tetramer ous",
                                 "tetraosulphate": "tetrao sulphate", "tgethr": "together",
                                 "thammana": "Tamannaah Bhatia", "that'd": "that would",
                                 "that'd've": "that would have", "that's": "that is", "thateasily": "that easily",
                                 "thausand": "thousand", "theeventchronicle": "the event chronicle",
                                 "theglobeandmail": "the globe and mail", "theguardian": "the guardian",
                                 "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                                 "thereanyone": "there anyone", "theuseof": "thereof", "they'd": "they would",
                                 "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                                 "they're": "they are", "they've": "they have", "theyreally": "they really",
                                 "thiests": "atheists",
                                 "thinkinh": "thinking", "thinkstrategic": "think strategic",
                                 "thinksurvey": "think survey", "this's": "this is",
                                 "thisAustralian": "this Australian", "thrir": "their", "timefram": "timeframe",
                                 "timejobs": "time jobs", "timeloop": "time loop", "timesence": "times ence",
                                 "timesjobs": "times jobs", "timesnow": "24-hour English news channel in India",
                                 "timesspark": "times spark", "timesup": "times up", "timetabe": "timetable",
                                 "timetraveling": "timet raveling", "timetraveller": "time traveller",
                                 "timetravelling": "timet ravelling", "timewaste": "time waste", "to've": "to have",
                                 "toastsexual": "toast sexual", "togofogo": "togo fogo", "tonogenesis": "tone",
                                 "toostupid": "too stupid", "toponymous": "top onymous", "tormentous": "torment ous",
                                 "torvosaurus": "Torosaurus", "totalinvestment": "total investment",
                                 "towayrds": "towards", "tradeplus": "trade plus",
                                 "trageting": "targeting", "tranfusions": "transfusions",
                                 "transexualism": "transsexualism", "transgenus": "trans genus",
                                 "transness": "trans gender",
                                 "transtrenders": "incredibly disrespectful to real transgender people",
                                 "trausted": "trusted", "treatens": "threatens", "treatmenent": "treatment",
                                 "treatmentshelp": "treatments help",
                                 "treetment": "treatment", "tremendeous": "tremendous",
                                 "triangleright": "triangle right", "tricompartmental": "tri compartmental",
                                 "trigonomatry": "trigonometry", "trillonere": "trillones",
                                 "trimegistus": "Trismegistus", "triphosphorus": "tri phosphorus",
                                 "trueoutside": "true outside", "trumpdating": "trump dating",
                                 "trumpers": "Trumpster", "trumpervotes": "trumper votes",
                                 "trumpites": "Trump supporters", "trumplies": "trump lies",
                                 "trumpology": "trump ology", "trustless": "t rustless", "trustworhty": "trustworthy",
                                 "trustworhy": "trustworthy", "tusaki": "tu saki", "tusami": "tu sami",
                                 "tusts": "trusts", "twiceusing": "twice using", "twinflame": "twin flame",
                                 "tyrannously": "tyrannous", "u.s.": "USA", "u.s.a": "USA", "u.s.a.": "USA",
                                 "ugggggggllly": "ugly", "uimovement": "ui movement", "umumoney": "umu money",
                                 "unacademy": "Unacademy", "unamendable": "un amendable",
                                 "unanonymously": "un anonymously", "unblacklisted": "un blacklisted",
                                 "uncosious": "uncopious", "uncouncious": "unconscious",
                                 "underdevelopement": "under developement", "undergraduation": "under graduation",
                                 "understamding": "understanding", "underthinking": "under thinking",
                                 "undervelopment": "undevelopment", "underworldly": "under worldly",
                                 "unergonomic": "un ergonomic", "unforgottable": "unforgettable",
                                 "uninstrusive": "unintrusive", "unkilled": "un killed", "unmoraled": "unmoral",
                                 "unpermissioned": "unper missioned", "unrightly": "un rightly",
                                 "upcomedians": "up comedians",
                                 "upwork": "up work", "urotone": "protone", "use38": "use", "usebase": "use base",
                                 "usedtoo": "used too", "usedvin": "used vin", "usefl": "useful",
                                 "userbags": "user bags", "userflows": "user flows", "usertesting": "user testing",
                                 "useul": "useful", "ushually": "usually", "usict": "USSCt", "uslme": "some",
                                 "uspset": "upset", "usucaption": "usu caption", "utilitas": "utilities",
                                 "utmterm": "utm term", "vairamuthus": "vairamuthu s", "vegetabale": "vegetable",
                                 "vegetablr": "vegetable", "vegetarean": "vegetarian", "vegetaries": "vegetables",
                                 "venetioned": "Venetianed", "vertigos": "vertigo s", "vetronus": "verrons",
                                 "vibhushant": "vibhushan t", "vicevice": "vice vice", "virituous": "virtuous",
                                 "visahouse": "visa house",
                                 "vitilgo": "vitiligo", "vivipoarous": "viviparous", "volime": "volume",
                                 "votebanks": "vote banks", "wanket": "wanker", "wantrank": "want rank",
                                 "washingtontimes": "washington times", "wasn't": "was not", "watchtime": "watch time",
                                 "wattman": "watt man",
                                 "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                                 "we'll've": "we will have", "we're": "we are", "we've": "we have",
                                 "weatern": "western", "webdevelopement": "web developement", "webmusic": "web music",
                                 "weenus": "elbow skin",
                                 "weightwithout": "weight without", "welcomemarriage": "welcome marriage",
                                 "welcomeromanian": "welcome romanian", "wellsfargoemail": "wellsfargo email",
                                 "weren't": "were not", "westsouth": "west south", "what'll": "what will",
                                 "what'll've": "what will have", "what're": "what are", "what's": "what is",
                                 "what've": "what have", "whatasapp": "WhatsApp", "whatcus": "what cause",
                                 "whatshapp": "WhatsApp", "whatsupp": "WhatsApp", "whattsup": "WhatsApp",
                                 "wheatestone": "wheatstone", "whemever": "whenever", "when's": "when is",
                                 "when've": "when have",
                                 "where Burkhas": "wear Burqas", "where'd": "where did", "where's": "where is",
                                 "where've": "where have", "whilemany": "while many", "whitegirls": "white girls",
                                 "whiteheds": "whiteheads", "whitelash": "white lash",
                                 "whitesplaining": "white splaining", "whitetning": "whitening",
                                 "whitewalkers": "white walkers", "who'll": "who will", "who'll've": "who will have",
                                 "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                                 "will've": "will have", "willhappen": "will happen", "winor": "win",
                                 "withgoogle": "with google", "withoutcheck": "without check",
                                 "withoutregistered": "without registered", "withoutyou": "without you",
                                 "womansplained": "womans plained", "womansplaining": "wo mansplaining",
                                 "womenizer": "womanizer", "won't": "will not", "won't've": "will not have",
                                 "workaway": "work away",
                                 "workdone": "work done", "workouses": "workhouses", "workperson": "work person",
                                 "worldbusiness": "world business", "worldmax": "wholesaler of drum parts",
                                 "worldquant": "world quant", "worldrank": "world rank",
                                 "worldwideley": "worldwide ley", "worstplatform": "worst platform",
                                 "would've": "would have",
                                 "wouldd": "would", "wouldn't": "would not", "wouldn't've": "would not have",
                                 "wowmen": "women", "ww 1": " WW1 ", "ww 2": " WW2 ", "y'all": "you all",
                                 "y'all'd": "you all would", "y'all'd've": "you all would have",
                                 "y'all're": "you all are",
                                 "y'all've": "you all have", "yaerold": "year old", "yeardold": "years old",
                                 "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                                 "you'll've": "you will have", "you're": "you are", "you've": "you have",
                                 "youbecome": "you become",
                                 "youbever": "you bever", "yousician": "musician", "zenfone": "zen fone",
                                 "zigolo": "gigolo", "zoneflex": "zone flex", "zymogenous": "zymogen ous",
                                 "ʙᴏᴛtoᴍ": "bottom", "ι": "i", "υ": "u", "в": "b",
                                 "м": "m", "н": "h", "т": "t", "ѕ": "s", "ᴀ": "a", "™": "trade mark", "∠bad": "bad",
                                 "」come": "come", "操你妈": "fuck your mother", "走go": "go",
                                 "😀": "stuck out tongue", "😂": "joy", "😉": "wink", }

        self.contraction_dict_lower = {" coinvest ": " invest ", " documentarie ": " documentaries ",
                                       " don t ": " do not ", " dont ": " do not ", " dsire ": " desire ",
                                       " govermen ": "goverment", " hinus ": " hindus ", " im ": " i am ",
                                       " incect ": " insect ",
                                       " internshala ": " internship and online training platform in india ",
                                       " jusification ": "justification", " justificatio ": "justification",
                                       " padmaavat ": " padmavati ", " padmaavati ": " padmavati ",
                                       " padmavat ": " padmavati ", " racious ": "discrimination expression of racism",
                                       " righten ": " tighten ", " s.p ": " ", " u ": " you ", " u r ": " you are ",
                                       " u.k ": " uk ", " u.s ": "usa", " yr old ": " years old ", "#": "",
                                       "#metoo": "metoo", "'cause": "because", "...": ".",
                                       "23 andme": "privately held personal genomics and biotechnology company in california",
                                       "2fifth": "twenty fifth", "2fourth": "twenty fourth",
                                       "2nineth": "twenty nineth", "2third": "twenty third", "4fifth": "forty fifth",
                                       "4fourth": "forty fourth", "4sixth": "forty sixth", "@ usafmonitor": "",
                                       "a**": "ass", "a****": "assho", "abandined": "abandoned", "abandonee": "abandon",
                                       "abcmouse": "abc mouse", "abdimen": "abdomen", "abhimana": "abhiman a",
                                       "abnegetive": "abnegative", "abonymously": "anonymously", "abussive": "abusive",
                                       "accedentitly": "accidentally", "accomany": "accompany",
                                       "accompishments": "accomplishments", "accomplihsments": "accomplishments",
                                       "accomplishmments": "accomplishments", "accompliushments": "accomplishments",
                                       "accomploshments": "accomplishments", "accomplsihments": "accomplishments",
                                       "accompplishments": "accomplishments", "accusition": "accusation",
                                       "achecomes": "ache comes", "achiements": "achievements",
                                       "achievership": "achievers hip", "achivenment": "achievement",
                                       "achviements": "achievements", "acidtone": "acid tone",
                                       "activationenergy": "activation energy", "activemoney": "active money",
                                       "acumens": "acumen s", "addmovement": "add movement", "adgestment": "adjustment",
                                       "adhaar": "adhara", "adsresses": "address", "adstargets": "ads targets",
                                       "advamcements": "advancements", "advantegeous": "advantageous",
                                       "aesexual": "asexual", "aevelopment": "development", "afircans": "africans",
                                       "africaget": "africa get", "afterafterlife": "after afterlife",
                                       "afvertisements": "advertisements", "aggeement": "agreement",
                                       "agoraki": "ago raki",
                                       "agreementindia": "agreement india", "ahemadabad": "ahmadabad",
                                       "ahmdabad": "ahmadabad", "ahmedbad": "ahmedabad", "ain't": "is not",
                                       "airindia": "air india", "aknowlege": "knowledge", "alahabad": "allahabad",
                                       "alamotyrannus": "alamo tyrannus", "alggorithms": "algorithms",
                                       "algorithimic": "algorithmic", "algorithom": "algorithm",
                                       "algoritmic": "algorismic", "algorthmic": "algorithmic",
                                       "algortihms": "algorithms", "alimoney": "alimony", "allhabad": "allahabad",
                                       "allmang": "all mang", "alogorithms": "algorithms",
                                       "alwayshired": "always hired",
                                       "amedabad": "ahmedabad", "amenclinics": "amen clinics",
                                       "amendmending": "amend mending", "amenitycompany": "amenity company",
                                       "americanmedicalassoc": "american medical assoc", "amette": "annette",
                                       "amharc": "amarc", "amigofoods": "amigo foods", "ammusement": "amusement",
                                       "amn't": "is not",
                                       "amorphus": "amorph us", "anarold": "android", "andkyokushin": "and kyokushin",
                                       "androneda": "andromeda", "anglosaxophone": "anglo saxophone",
                                       "animlaistic": "animalistic", "anmuslim": "an muslim", "annilingus": "anilingus",
                                       "annoincement": "announcement", "annonimous": "anonymous",
                                       "anomymously": "anonymously", "anonimously": "anonymously",
                                       "anonoymous": "anonymous", "antergos": "anteros",
                                       "anti cipation": "anticipation", "anti-semitic": "anti-semitic",
                                       "antibigots": "anti bigots", "antibrahmin": "anti brahmin",
                                       "anticancerous": "anti cancerous", "anticoncussive": "anti concussive",
                                       "antihindus": "anti hindus", "antilife": "anti life", "antimeter": "anti meter",
                                       "antireligion": "anti religion", "antivirius": "antivirus",
                                       "antritrust": "antitrust", "anxietymake": "anxiety make",
                                       "anyonestudied": "anyone studied", "aomeone": "someone",
                                       "apartheidisrael": "apartheid israel",
                                       "apartmentfinder": "apartment finder", "appsmoment": "apps moment",
                                       "aquous": "aqueous", "arangodb": "arango db", "arceius": "arcesius",
                                       "arceous": "araceous", "archaemenid": "achaemenid", "are'nt": "are not",
                                       "arectifier": "rectifier", "aremenian": "armenian",
                                       "aren't": "are not", "arencome": "aren come", "arewhatsapp": "are whatsapp",
                                       "argruments": "arguments", "argumentetc": "argument etc",
                                       "aristocracylifestyle": "aristocracy lifestyle", "armanents": "armaments",
                                       "armenains": "armenians", "armenized": "armenize", "arn't": "are not",
                                       "aroused21000": "aroused 21000", "arragent": "arrogant", "asexaul": "asexual",
                                       "ashifa": "asifa", "asianisation": "becoming asia", "asiasoid": "asian",
                                       "assignmentcanyon": "assignment canyon", "aswell": "as well",
                                       "athiust": "athirst", "atlous": "atrous",
                                       "atonomous": "autonomous", "atrocitties": "atrocities",
                                       "attirements": "attire ments", "attrected": "attracted",
                                       "audetteknown": "audette known", "augumentation": "argumentation",
                                       "augumented": "augmented", "august2017": "august 2017", "aulphate": "sulphate",
                                       "auragabad": "aurangabad",
                                       "austentic": "austenitic", "austertana": "auster tana",
                                       "austinlizards": "austin lizards", "austira": "australia",
                                       "australia9": "australian", "australianism": "australian ism",
                                       "australua": "australia", "austrelia": "australia",
                                       "austrialians": "australians", "austrolia": "australia",
                                       "austroloid": "australoid", "austronauts": "astronauts",
                                       "autoliker": "auto liker", "autolikers": "auto likers",
                                       "autosexual": "auto sexual", "auwe": "wonder", "avacous": "vacuous",
                                       "avegeta": "ave geta", "awrong": "aw rong", "ayahusca": "ayahausca",
                                       "aysnchronous": "asynchronous", "aɴᴅ": "and", "aᴛ": "at", "b**": "bit",
                                       "b***": "bitc", "b****": "bitch", "b*ll": "bull", "b*tch": "bitch",
                                       "babadook": "a horror drama film", "babymust": "baby must",
                                       "bacteries": "batteries",
                                       "badaganadu": "brahmin community that mainly reside in karnataka",
                                       "badeffects": "bad effects", "badgermoles": "enormous, blind mammal",
                                       "badminaton": "badminton", "badmothing": "badmouthing",
                                       "badonicus": "sardonicus", "badtameezdil": "badtameez dil",
                                       "balanoglossus": "balanoglossus", "balckwemen": "balck women",
                                       "ballonets": "ballo nets", "ballyrumpus": "bally rumpus",
                                       "baloochistan": "balochistan", "banggood": "bang good", "bapus": "bapu s",
                                       "basedstickman": "based stickman", "batmanvoice": "batman voice",
                                       "batsmencould": "batsmen could", "batteriesplus": "batteries plus",
                                       "beacsuse": "because",
                                       "becausr": "because", "becomesdouble": "becomes double", "becone": "become",
                                       "beerus": "the god of destruction", "belagola": "bela gola",
                                       "beloney": "boloney", "beltholders": "belt holders",
                                       "beluchistan": "balochistan", "bengolis": "bengalis", "betterdtu": "better dtu",
                                       "betterindia": "better india", "betterv3": "better",
                                       "betweenmanagement": "between management", "bhagwanti": "bhagwant i",
                                       "bhlushes": "blushes", "bhraman": "braman", "bhushi": "thushi", "bi*ch": "bitch",
                                       "bigly": "big league", "bigolive": "big olive",
                                       "bindusar": "bind usar", "bingobox": "an investment company",
                                       "biromantical": "bi romantical", "bit*h": "bitch", "bitc*": "bitch",
                                       "bitcjes": "bitches", "bittergoat": "bitter goat", "blackbeauty": "black beauty",
                                       "blackboxing": "black boxing", "blackdotes": "black dotes",
                                       "blackmarks": "black marks", "blackmoney": "black money",
                                       "blackpaper": "black paper", "blackphone": "black phone",
                                       "blackphones": "black phones", "blackpink": "black pink",
                                       "blackpower": "black power", "blackwashing": "black washing",
                                       "blindfoldedly": "blindfolded", "blockchains": "blockchain",
                                       "blocktime": "block time", "bodhidharman": "bodhidharma",
                                       "bogosort": "bogo sort", "boldspot": "bolds pot", "bonechiller": "bone chiller",
                                       "bonesetters": "bone setters", "bonespur": "bones pur", "book'em": "book them",
                                       "boothworld": "booth world", "bootyplus": "booty plus",
                                       "bootythongs": "booty thongs", "borokabama": "barack obama",
                                       "boruto": "naruto next generations", "brahmanwad": "brahman wad",
                                       "brahmanwadi": "brahman wadi", "brainsway": "brains way",
                                       "bramanical": "brahmanical", "bramanistic": "brahmanistic",
                                       "brexit": "british exit", "brightiest": "brightest",
                                       "brightspace": "brights pace", "brlieve": "believe", "brophytes": "bryophytes",
                                       "brotherzone": "brother zone", "brotherzoned": "brother zoned",
                                       "brusssels": "brussels", "budgetair": "budget air", "bugdget": "budget",
                                       "bugetti": "bugatti", "bushiri": "bushire",
                                       "businessbor": "business bor", "businessinsider": "business insider",
                                       "businiss": "business", "busscas": "buscas", "bussiest": "fussiest",
                                       "bussinessmen": "businessmen", "bussinss": "bussings", "bustees": "bus tees",
                                       "c'mon": "come on", "c'mooooooon": "come on",
                                       "caausing": "causing", "cab't": "can not", "cakemaker": "cake maker",
                                       "calcalus": "calculus", "calccalculus": "calc calculus", "calculus1": "calculus",
                                       "caligornia": "california", "cammando": "commando", "camponente": "component",
                                       "campusthrough": "campus through",
                                       "campuswith": "campus with", "can't": "cannot", "canmuslims": "can muslims",
                                       "cann't": "can not", "cant't": "can not", "canterlever": "canter lever",
                                       "capesindia": "capes india", "capuletwant": "capulet want",
                                       "careerplus": "career plus",
                                       "careongo": "india first and largest online distributor of medicines",
                                       "carnivorus": "carnivorous", "carnivourous": "carnivorous",
                                       "casterating": "castrating", "catagorey": "category", "catallus": "catullus",
                                       "categoried": "categories", "catrgory": "category", "catwgorized": "categorized",
                                       "cau.sing": "causing", "causians": "crusians",
                                       "causinng": "causing", "ceftriazone": "ceftriaxone", "celcious": "delicious",
                                       "celemente": "clemente", "celibatess": "celibates", "cellsius": "celsius",
                                       "celsious": "cesious", "cemenet": "cement", "centimiters": "centimeters",
                                       "cetusplay": "cetus play",
                                       "changetip": "change tip", "chaparone": "chaperone",
                                       "chapterwise": "chapter wise", "checkusers": "check users",
                                       "cheekboned": "cheek boned", "chestbusters": "chest busters",
                                       "cheverlet": "cheveret", "chickengonia": "chicken gonia",
                                       "chigoe": "sub-tropical climates", "chinawares": "china wares",
                                       "chinesese": "chinese", "chiropractorone": "chiropractor one",
                                       "chlamydomanas": "chlamydomonas", "chromecast": "chrome cast",
                                       "chromonema": "chromo nema", "chronexia": "chronaxia", "chylus": "chylous",
                                       "circumradius": "circum radius", "ciswomen": "cis women",
                                       "citiesbetter": "cities better",
                                       "citruspay": "citrus pay", "cityairbus": "city airbus", "clause55": "clause",
                                       "claustophobic": "claustrophobic", "clearworld": "clear world",
                                       "clevercoyote": "clever coyote", "cloudways": "cloud ways",
                                       "clusture": "culture", "cobditioners": "conditioners", "cobustion": "combustion",
                                       "coclusion": "conclusion", "codeagon": "code agon",
                                       "coincedences": "coincidences", "coincidents": "coincidence",
                                       "coinfirm": "confirm", "coinsidered": "considered", "coinsized": "coin sized",
                                       "coinstop": "coins top", "cointries": "countries", "cointry": "country",
                                       "colmbus": "columbus", "comandant": "commandant", "come2": "come to",
                                       "comepleted": "completed", "cometitive": "competitive",
                                       "comfortzone": "comfort zone", "comlementry": "complementary",
                                       "commencial": "commercial", "commensalisms": "commensal isms",
                                       "commentsafe": "comment safe",
                                       "commissiioned": "commissioned", "commissionerates": "commissioner ates",
                                       "commissionets": "commissioners", "commudus": "commodus", "comodies": "corodies",
                                       "compartmentley": "compartment", "complusary": "compulsory",
                                       "componendo": "compon endo", "con't": "can not", "conartist": "con-artist",
                                       "conciousnes": "conciousness", "conciousnesss": "consciousnesses",
                                       "conclusionless": "conclusion less", "concuous": "conscious",
                                       "condioner": "conditioner", "condtioners": "conditioners",
                                       "conectiin": "connection", "conectu": "connect you", "conents": "contents",
                                       "confiment": "confident",
                                       "confousing": "confusing", "confundus": "con fundus", "congoid": "congolese",
                                       "congressi": "congress", "congusion": "confusion", "conpartment": "compartment",
                                       "consciousuness": "consciousness", "consciousx5": "conscious",
                                       "conscoiusness": "consciousness", "consicious": "conscious",
                                       "constiously": "consciously",
                                       "constitutionaldevelopment": "constitutional development",
                                       "contestious": "contentious", "contiguious": "contiguous",
                                       "contraproductive": "contra productive", "cooerate": "cooperate",
                                       "cooktime": "cook time", "cooldige": "the 30th president of the united states",
                                       "cooldrink": "cool drink", "coolmuster": "cool muster",
                                       "coreligionist": "co religionist", "corpgov": "corp gov",
                                       "corypheus": "coryphees", "cosicous": "conscious", "coud've": "could have",
                                       "could'nt": "could not", "could've": "could have", "couldn't": "could not",
                                       "counciousness": "conciousness", "countinous": "continuous",
                                       "countryball": "country ball", "countryhow": "country how",
                                       "countryless": "having no country", "countrypeople": "country people",
                                       "cousan": "cousin", "cousera": "couler", "cousi ": "cousin ",
                                       "covfefe": "coverage", "cox": "cox", "cracksbecause": "cracks because",
                                       "crazymaker": "crazy maker", "creditdation": "accreditation",
                                       "cretinously": "cretinous", "cryptochristians": "crypto christians",
                                       "cryptoworld": "crypto world", "cuckerberg": "zuckerberg",
                                       "cuckholdry": "cuckold", "culturr": "culture", "cushelle": "cush elle",
                                       "cusjo": "cusco",
                                       "customshoes": "customs hoes", "customzation": "customization",
                                       "cutaneously": "cu taneously", "cyclohexenone": "cyclohexanone", "d***": "dick",
                                       "daigonal": "diagonal", "dailycaller": "daily caller", "danergous": "dangerous",
                                       "dangergrous": "dangerous", "dangorous": "dangerous",
                                       "dardandus": "dardanus", "darkweb": "dark web",
                                       "darkwebcrawler": "dark webcrawler", "darskin": "dark skin", "ddn't": "did not",
                                       "dealignment": "de alignment", "deangerous": "dangerous",
                                       "deartment": "department", "deathbecomes": "death becomes",
                                       "defpush": "def push",
                                       "degenarate": "degenerate", "degenerous": "de generous",
                                       "deindustralization": "deindustrialization", "dejavus": "dejavu s",
                                       "demanters": "decanters", "demantion": "detention", "demigirl": "demi girl",
                                       "demihuman": "demi human", "demisexuality": "demi sexuality",
                                       "demisexuals": "demisexual",
                                       "democratizationed": "democratization ed", "demogorgan": "demogorgon",
                                       "demonentization": "demonetization", "demonetisation": "demonetization",
                                       "demonetised": "demonetized", "demonetistaion": "demonetization",
                                       "demonetizations": "demonetization", "demonetize": "demonetized",
                                       "demonetizedd": "demonetized", "demonetizing": "demonetized",
                                       "denomenator": "denominator", "departmenthow": "department how",
                                       "deplorables": "deplorable", "deployements": "deployments",
                                       "depolyment": "deployment", "dermatillomaniac": "dermatillomania",
                                       "desemated": "decimated", "deshabhimani": "desha bhimani",
                                       "designation-": "designation", "designbold": "online photo editor design studio",
                                       "determenism": "determinism", "determenistic": "deterministic",
                                       "develepoments": "developments", "developmenent": "development",
                                       "developmeny": "development", "devendale": "evendale", "deverstion": "diversion",
                                       "devlelpment": "development", "devopment": "development", "di**": "dick",
                                       "dictioneries": "dictionaries", "didn't": "did not",
                                       "didn't_work": "did not work", "didn`t": "did not", "diffenky": "differently",
                                       "diffussion": "diffusion", "difigurement": "disfigurement", "difusse": "diffuse",
                                       "digitizeindia": "digitize india", "digonal": "di gonal",
                                       "digonals": "diagonals", "digustingly": "disgustingly", "diiscuss": "discuss",
                                       "dilusion": "delusion", "dimensionalise": "dimensional ise",
                                       "dimensiondimensions": "dimension dimensions", "dimensonless": "dimensionless",
                                       "dimensons": "dimensions", "dingtone": "ding tone", "dioestrus": "di oestrus",
                                       "diphosphorus": "di phosphorus", "diplococcus": "diplo coccus",
                                       "disagreemen": "disagreement", "disagreementt": "disagreement",
                                       "disagreementts": "disagreements", "disangagement": "disengagement",
                                       "disastrius": "disastrous", "disclousre": "disclosure",
                                       "discrimantion": "discrimination", "discriminatein": "discrimination",
                                       "disengenuously": "disingenuously", "dishonerable": "dishonorable",
                                       "disngush": "disgust", "disquss": "discuss", "disscused": "discussed",
                                       "disulphate": "di sulphate", "dn't": "do not", "do'nt": "do not",
                                       "doccuments": "documents", "docoment": "document",
                                       "doctrne": "doctrine", "docuements": "documents", "docyments": "documents",
                                       "does'nt": "does not", "does't": "does not", "doesen't": "does not",
                                       "doesgauss": "does gauss", "doesn't": "does not", "dogmans": "dogmas",
                                       "dogooder": "do gooder",
                                       "dolostones": "dolostone", "don''t": "do not", "don't": "do not",
                                       "don'tcare": "do not care", "don'ts": "do not",
                                       "donaldtrumping": "donald trumping", "doneclaim": "done claim",
                                       "donedone": "done done", "donesnt": "doesnt", "dont't": "do not",
                                       "dood-": "dood", "dormmanu": "dormant", "dose't": "does not",
                                       "doublelife": "double life", "dowsn't": "does not", "dpn't": "do not",
                                       "draconius": "draconis", "dragonchain": "dragon chain",
                                       "dragonflag": "dragon flag", "dragonglass": "dragon glass",
                                       "dragonkeeper": "dragon keeper", "dragonknight": "dragon knight",
                                       "dramafever": "drama fever", "dream11": " fantasy sports platform in india ",
                                       "drparment": "department", "drumpf": "trump", "dsymenorrhoea": "dysmenorrhoea",
                                       "dumbassess": "dumbass", "duramen": "dura men", "durgahs": "durgans",
                                       "dushasana": "dush asana", "dusra": "dura", "dusyant": "distant",
                                       "e.g.": "for example", "earvphone": "earphone",
                                       "easedeverything": "eased everything", "eauipments": "equipments",
                                       "ecoworld": "eco world", "ecumencial": "ecumenical", "eduacated": "educated",
                                       "egomanias": "ego manias", "electronegativty": "electronegativity",
                                       "electroneum": "electro neum", "elementsbond": "elements bond",
                                       "eletronegativity": "electronegativity", "elinment": "eloinment",
                                       "elitmus": "indian company that helps companies in hiring employees",
                                       "embaded": "embased", "emellishments": "embellishments", "emiratis": "emirates",
                                       "emmanvel": "emmarvel", "emmenagogues": "emmenagogue",
                                       "emouluments": "emoluments", "empericus": "imperious", "emplement": "implement",
                                       "employmentnews": "employment news", "employmment": "employment",
                                       "enchacement": "enchancement", "enchroachment": "encroachment",
                                       "encironment": "environment",
                                       "encironmental": "environmental", "endfragment": "end fragment",
                                       "endrosment": "endorsement", "eneyone": "anyone", "engangement": "engagement",
                                       "enivironment": "environment", "enjoiment": "enjoyment",
                                       "enliightenment": "enlightenment", "enrollnment": "enrollment",
                                       "entaglement": "entanglement",
                                       "entartaiment": "entertainment", "enthusiasmless": "enthusiasm less",
                                       "entitlments": "entitlements", "entusiasta": "enthusiast",
                                       "entwicklungsroman": "entwicklungs roman", "envinment": "environment",
                                       "enviorement": "environment", "envioronments": "environments",
                                       "envirnmetalists": "environmentalists", "environmentai": "environmental",
                                       "envisionment": "envision ment", "epicodus": "episodes",
                                       "ergophobia": "ergo phobia", "ergosphere": "ergo sphere",
                                       "ergotherapy": "ergo therapy", "ernomous": "enormous",
                                       "errounously": "erroneously", "essaytyper": "essay typer", "et.al": "elsewhere",
                                       "ethethnicitesnicites": "ethnicity",
                                       "eurooe": "europe", "ev'rybody": "everybody", "evangilitacal": "evangelical",
                                       "evenafter": "even after", "eventhogh": "even though", "everbodys": "everybody",
                                       "everperson": "ever person", "everydsy": "everyday", "everyfirst": "every first",
                                       "everyo0 ne": "everyone",
                                       "everyonediestm": "everyonedies tm", "everyonr": "everyone",
                                       "examsexams": "exams exams", "exause": "excuse", "exclusinary": "exclusionary",
                                       "excritment": "excitement", "exilarchate": "exilarch ate",
                                       "exmuslims": "ex-muslims", "experimently": "experiment",
                                       "expertthink": "expert think",
                                       "expessway": "expressway", "exserviceman": "ex serviceman",
                                       "exservicemen": "ex servicemen", "extemporeneous": "extemporaneous",
                                       "extraterestrial": "extraterrestrial", "extroneous": "extraneous",
                                       "exust": "ex ust", "eyemake": "eye make", "f**": "fuc", "f***": "fuck",
                                       "f**k": "fuck", "faceplusplus": "face plusplus", "faegon": "wagon",
                                       "faidabad": "faizabad", "fakenews": "fake news", "falaxious": "fallacious",
                                       "fallicious": "fallacious", "faminazis": "feminazis",
                                       "famousbirthdays": "famous birthdays", "fanessay": "fan essay",
                                       "fanmenow": "fan menow", "fantocone": "fantocine", "farmention": "farm ention",
                                       "farmhousebistro": "farmhouse bistro", "fecetious": "facetious",
                                       "feelinfs": "feelings", "feelingstupid": "feeling stupid",
                                       "feelomgs": "feelings", "feelonely": "feel onely", "feelwhen": "feel when",
                                       "femanists": "feminists", "femenise": "feminise", "femenism": "feminism",
                                       "femenists": "feminists", "feminisam": "feminism",
                                       "feminists": "feminism supporters", "fentayal": "fentanyl",
                                       "fergussion": "percussion", "fermentqtion": "fermentation",
                                       "fernbus": "fern bus",
                                       "fevelopment": "development", "fewshowanyremorse": "few show any remorse",
                                       "feymann": "heymann", "filabustering": "filibustering",
                                       "findamental": "fundamental",
                                       "fingols": "finnish people are supposedly descended from mongols",
                                       "firdausiya": "firdausi ya", "fisgeting": "fidgeting", "fislike": "dislike",
                                       "fitgirl": "fit girl",
                                       "flashgive": "flash give", "flemmings": "flemming", "flixbus": "flix bus",
                                       "followingorder": "following order", "fondels": "fondles",
                                       "fonecare": "f onecare", "foodlovee": "food lovee", "forcestop": "forces top",
                                       "forgetfulnes": "forgetfulness", "forgottenfaster": "forgotten faster",
                                       "français": "france", "fraysexual": "fray sexual", "freemansonry": "freemasonry",
                                       "freshersworld": "freshers world", "freus": "frees", "friedzone": "fried zone",
                                       "frighter": "fighter", "fromgermany": "from germany", "fronend": "friend",
                                       "frosione": "erosion",
                                       "frozen tamod": "pornographic website", "frustratd": "frustrate", "fu.k": "fuck",
                                       "fuckboy": "fuckboy", "fuckboys": "fuckboy", "fuckgirl": "fuck girl",
                                       "fuckgirls": "fuck girls", "fucktrumpet": "fuck trumpet",
                                       "fudamental": "fundamental", "fundamantal": "fundamental",
                                       "fundemantally": "fundamentally", "fundsindia": "funds india",
                                       "fusanosuke": "fu sanosuke", "fusiondrive": "fusion drive",
                                       "fusiongps": "fusion gps", "führer": "fuhrer", "g'bye": "goodbye",
                                       "g'morning": "good morning", "g'night": "goodnight", "gadagets": "gadgets",
                                       "gadgetpack": "gadget pack", "gadgetsnow": "gadgets now",
                                       "galastop": "galas top", "gamenights": "game nights", "gamergaye": "gamersgate",
                                       "gamersexual": "gamer sexual", "gangstalkers": "gangs talkers",
                                       "gapbuster": "gap buster", "garthago": "carthago", "garycrum": "gary crum",
                                       "gaudry - schost": "", "gaushalas": "gaus halas",
                                       "gayifying": "gayed up with homosexual love", "gayke": "gay online retailers",
                                       "geerymandered": "gerrymandered", "geev'um": "give them", "geftman": "gentman",
                                       "geminatus": "geminates", "genderfluid": "gender fluid",
                                       "genetilia": "genitalia",
                                       "geniusses": "geniuses", "genocidizing": "genociding", "genuiuses": "geniuses",
                                       "geomentrical": "geometrical", "geramans": "germans",
                                       "germanised": "german ised", "germanity": "german",
                                       "germanized": "become german", "germanophone": "germanophobe",
                                       "germanyl": "germany l",
                                       "germeny": "germany", "get1000": "get 1000", "get630": "get 630", "get90": "get",
                                       "getallparts": "get allparts", "getdepersonalization": "get depersonalization",
                                       "getdrip": "get drip", "getfinancing": "get financing", "getile": "gentile",
                                       "getmy": "get my",
                                       "getsmuggled": "get smuggled", "gettibg": "getting", "gettubg": "getting",
                                       "gevernment": "government", "ghazibad": "ghazi bad", "ghumendra": "bhupendra",
                                       "girlfrnd": "girlfriend", "girlriend": "girlfriend", "give'em": "give them",
                                       "glocuse": "louse",
                                       "glrous": "glorious", "glueteus": "gluteus", "goalwise": "goal wise",
                                       "goatnuts": "goat nuts", "gobackmodi": "goback modi",
                                       "goegraphies": "geographies", "gogoro": "gogo ro", "golddig": "gold dig",
                                       "goldendict": "open-source dictionary program", "goldengroup": "golden group",
                                       "goldmedal": "gold medal", "goldmont": "microarchitecture in intel",
                                       "goldquest": "gold quest", "golemized": "polemized", "golusu": "gol usu",
                                       "gomovies": "go movies", "gonverment": "government", "goodfirms": "good firms",
                                       "goodfriends": "good friends", "goodspeaks": "good speaks",
                                       "google4": "google", "googlemapsapi": "googlemaps api",
                                       "googology": "online encyclopedia", "googolplexain": "googolplexian",
                                       "gorakpur": "gorakhpur", "gorgops": "gorgons", "gorichen": "gori chen mountain",
                                       "gorlfriend": "girlfriend", "got7": "got", "gothras": "goth ras",
                                       "gouing": "going", "goundar": "gondar", "gouverments": "governments",
                                       "goverenments": "governments", "govermenent": "government",
                                       "governening": "governing", "governmaent": "government",
                                       "govnments": "movements", "govrment": "government",
                                       "govshutdown": "gov shutdown",
                                       "govtribe": "provides real-time federal contracting market intel",
                                       "gpusli": "gusli", "grab'em": "grab them", "greatuncle": "great uncle",
                                       "greenhouseeffect": "greenhouse effect",
                                       "greenseer": "people who possess the magical ability",
                                       "greysexual": "grey sexual", "gridgirl": "female models of the race",
                                       "guardtime": "guard time", "gujaratis": "gujarati",
                                       "gujjus": "derogatory gujarati", "gujratis": "gujarati", "gusdur": "gus dur",
                                       "gusenberg": "gutenberg", "gyroglove": "wearable technology", "gʀᴇat": "great",
                                       "h***": "hole", "h*ck": "hack", "habius - corpus": "habeas corpus",
                                       "hackerrank": "hacker rank",
                                       "haden't": "had not", "hadn't": "had not", "haffmann": "hoffmann",
                                       "halfgirlfriend": "half girlfriend", "handgloves": "hand gloves",
                                       "hanfus": "hannus", "haoneymoon": "honeymoon", "haraasment": "harassment",
                                       "harashment": "harassment", "harasing": "harassing",
                                       "harassd": "harassed", "harassument": "harassment", "harbaceous": "herbaceous",
                                       "hasanyone": "has anyone", "hasidus": "hasid us", "hasn't": "has not",
                                       "hasserment": "harassment", "hatrednesss": "hatred", "haufman": "kaufman",
                                       "haven't": "have not",
                                       "havn't": "have not", "he''s": "he is", "he'd": "he would", "he'll": "he will",
                                       "he's": "he is", "heeadphones": "headphones",
                                       "heightism": "height discrimination", "heineous": "heinous", "heitus": "leitus",
                                       "helisexual": "sexual",
                                       "hellochinese": "hello chinese", "here's": "here is", "heriones": "heroines",
                                       "hetereosexual": "heterosexual", "heteromantic": "hete romantic",
                                       "heteroromantic": "hetero romantic", "hetorosexuality": "hetoro sexuality",
                                       "hetorozygous": "heterozygous", "hetrogenous": "heterogenous",
                                       "hexanone": "hexa none",
                                       "hhhow": "how", "hidusim": "hinduism", "highschoold": "high school",
                                       "hillum": "helium", "himanity": "humanity", "himdus": "hindus",
                                       "hindhus": "hindus", "hindian": "north indian", "hindians": "north indian",
                                       "hindusm": "hinduism",
                                       "hindustanis": "", "hindusthanis": "hindustanis",
                                       "hippocratical": "hippocratical", "hkust": "hust", "hold'um": "hold them",
                                       "hollcaust": "holocaust", "holocause": "holocaust", "holocaustable": "holocaust",
                                       "holocaustal": "holocaust", "holocausting": "holocaust ing",
                                       "holocoust": "holocaust", "homageneous": "homogeneous", "homeseek": "home seek",
                                       "homogeneus": "homogeneous", "homologus": "homologous",
                                       "homoromantic": "homo romantic", "homosexualtiy": "homosexuality",
                                       "homosexulity": "homosexuality", "homosporous": "homos porous",
                                       "honeyfund": "honey fund",
                                       "honorkillings": "honor killings", "hopstop": "hops top",
                                       "hotelmanagement": "hotel management", "hotstar": "hot star",
                                       "housban": "housman", "householdware": "household ware",
                                       "housepoor": "house poor", "how'd": "how did", "how'd'y": "how do you",
                                       "how'll": "how will",
                                       "how's": "how is", "howddo": "how do", "howeber": "however",
                                       "howhat": "how that", "howknow": "howk now", "howlikely": "how likely",
                                       "howmaney": "how maney", "howwould": "how would", "htderabad": "hyderabad",
                                       "htmlutmterm": "html utm term",
                                       "hue mungus": "feminist bait", "huemanity": "humanity",
                                       "hugh mungus": "feminist bait", "hukous": "humous",
                                       "humancoalition": "human coalition", "humanfemale": "human female",
                                       "humanitariarism": "humanitarianism", "humanpark": "human park",
                                       "humanzees": "human zees", "humilates": "humiliates",
                                       "hurrasement": "hurra sement", "huskystar": "hu skystar", "husnai": "hussar",
                                       "husoone": "huso one", "huswifery": "huswife ry", "hwhat": "what",
                                       "hydeabad": "hyderabad", "hypercubus": "hypercubes",
                                       "hyperfocusing": "hyper focusing", "hypersexualize": "hyper sexualize",
                                       "hyperthalamus": "hyper thalamus", "hypogonadic": "hypogonadia",
                                       "hypotenous": "hypogenous", "hystericus": "hysteric us", "i''ve": "i have",
                                       "i'd": "i would", "i'd've": "i would have", "i'don": "i do not",
                                       "i'dve": "i would have", "i'il": "i will",
                                       "i'l": "i will", "i'll": "i will", "i'll've": "i will have",
                                       "i'llbe": "i will be", "i'lll": "i will", "i'm": "i am", "i'ma": "i am a",
                                       "i'mm": "i am", "i'mma": "i am a", "i'v": "i have",
                                       "i've": "i have", "i'vemade": "i have made", "i'veposted": "i have posted",
                                       "i'veve": "i have", "i'vè": "i have", "i'μ": "i am", "i.e.": "in other words",
                                       "ibdustries": "industries", "icompus": "corpus", "icrement": "increment",
                                       "idon'tgetitatall": "i do not get it at all", "idonesia": "indonesia",
                                       "idoot": "idiot", "ignasius": "ignacius", "ignoramouses": "ignoramuses",
                                       "ijustdon'tthink": "i just do not think", "illigitmate": "illegitimate",
                                       "illiustrations": "illustrations", "illlustrator": "illustrator",
                                       "illustratuons": "illustrations",
                                       "ilusha": "ilesha", "ilustrating": "illustrating", "imaprtus": "impetus",
                                       "immplemented": "implemented", "improment": "impriment",
                                       "imrovment": "improvement", "imrpovement": "improvement",
                                       "inapropriately": "inappropriately", "inatrumentation": "instrumentation",
                                       "incestious": "incestuous",
                                       "increaments": "increments", "indans": "indians", "indemendent": "independent",
                                       "indemenity": "indemnity", "indianarmy": "indian army",
                                       "indianleaders": "indian leaders", "indiareads": "india reads",
                                       "indiarush": "india rush", "indiginious": "indigenous",
                                       "indigoflight": "indigo flight",
                                       "indominus": "in dominus", "indrold": "android", "indrustry": "industry",
                                       "indtrument": "instrument", "industdial": "industrial", "industey": "industry",
                                       "industion": "induction", "industiry": "industry",
                                       "industrailzed": "industrialized", "industrilaization": "industrialization",
                                       "industrilised": "industrialised", "industrilization": "industrialization",
                                       "industris": "industries", "industrybuying": "industry buying",
                                       "industrzgies": "industries", "indusyry": "industry", "infrctuous": "infectuous",
                                       "infridgement": "infringement", "infringiment": "infringement",
                                       "infrustructre": "infrastructure",
                                       "infussions": "infusions", "infustry": "industry", "ingenhousz": "ingenious",
                                       "inheritantly": "inherently", "innermongolians": "inner mongolians",
                                       "innerskin": "inner skin", "inpersonations": "impersonations",
                                       "inplementing": "implementing", "inpregnated": "in pregnant",
                                       "inquoraing": "inquiring",
                                       "inradius": "in radius", "insdians": "indians",
                                       "insectivourous": "insectivorous", "insemenated": "inseminated",
                                       "insipudus": "insipidus", "inspirus": "inspires", "instagate": "instigate",
                                       "instantanous": "instantaneous", "instantneous": "instantaneous",
                                       "instatanously": "instantaneously",
                                       "instrumenot": "instrument", "instrumentalizing": "instrument",
                                       "instrumention": "instrument ion", "instumentation": "instrumentation",
                                       "insustry": "industry", "integumen": "integument", "intelliegent": "intelligent",
                                       "intermate": "inter mating", "interpersonation": "inter personation",
                                       "intrasegmental": "intra segmental",
                                       "intrusivethoughts": "intrusive thoughts", "intuous": "virtuous",
                                       "intwrsex": "intersex", "inudustry": "industry", "inuslating": "insulating",
                                       "invement": "movement", "inverstment": "investment",
                                       "investmentguru": "investment guru", "invetsment": "investment",
                                       "inviromental": "environmental",
                                       "involvedinthe": "involved in the", "invovement": "involvement",
                                       "iodoketones": "iodo ketones", "iodopovidone": "iodo povidone",
                                       "iovercome": "i overcome", "iphone6": "iphone", "iphone7": "iphone",
                                       "iphone8": "iphone", "iphonex": "iphone", "irakis": "iraki",
                                       "irreputable": "reputation", "irrevershible": "irreversible", "is't": "is not",
                                       "isdhanbad": "is dhanbad", "iservant": "servant", "islamphobia": "islam phobia",
                                       "islamphobic": "islam phobic", "isn't": "is not", "isn`t": "is not",
                                       "isotones": "iso tones",
                                       "ispakistan": "is pakistan", "isthmuses": "isthmus es", "istop": "i stop",
                                       "isunforgettable": "is unforgettable", "it''s": "it is", "it'also": "it is also",
                                       "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                                       "it'll've": "it will have",
                                       "it's": "it is", "it`s": "it is", "itstead": "instead", "itylus": "i tylus",
                                       "iusso": "kusso", "iv'e": "i have", "ivanious": "avanious",
                                       "janewright": "jane wright", "janusgraph": "janus graph", "jarrus": "harrus",
                                       "javadiscussion": "java discussion", "jeaslous": "jealous",
                                       "jeesyllabus": "jee syllabus", "jeisus": "jesus", "jerusalsem": "jerusalem",
                                       "jeruselam": "jerusalem", "jerussalem": "jerusalem", "jetbingo": "jet bingo",
                                       "jewprofits": "jew profits", "jewsplain": "jews plain",
                                       "jigolo": "gigolo", "jionee": "jinnee", "jlius": "julius",
                                       "jobberman": "jobber man", "joboutlook": "job outlook",
                                       "joboutlooks": "job outlooks", "josephius": "josephus",
                                       "journalust": "journalist", "judisciously": "judiciously", "juilus": "julius",
                                       "jusify": "justify", "jusrlt": "just", "justbasic": "just basic",
                                       "justcheaptickets": "just cheaptickets", "justcreated": "just created",
                                       "justdile": "justice", "justforex": "just forex", "justiciaries": "justiciary",
                                       "justinkase": "justin kase", "justyfied": "justified",
                                       "justyfy": "justify", "kabadii": "kabaddi", "kalitake": "kali take",
                                       "kamikazis": "kamikazes", "karonese": "karo people indonesia",
                                       "kashmiristan": "kashmir", "katgmandu": "katmandu", "kaushika": "kaushik a",
                                       "keralapeoples": "kerala peoples", "kerataconus": "keratoconus",
                                       "keratokonus": "keratoconus", "keywordseverywhere": "keywords everywhere",
                                       "khushali": "khushal i", "kick'em": "kick them",
                                       "killedshivaji": "killed shivaji", "killikng": "killing",
                                       "killograms": "kilograms", "killyou": "kill you",
                                       "kim jong-un": "the president of north korea", "knowble": "knowle",
                                       "knowldage": "knowledge", "knowledeg": "knowledge", "knowledgd": "knowledge",
                                       "knowledgewoods": "knowledge woods", "knowlefge": "knowledge",
                                       "knowlegdeable": "knowledgeable", "knownprogramming": "known programming",
                                       "knowyouve": "know youve", "koncerned": "concerned", "kotatsus": "kotatsu s",
                                       "kouldn't": "could not", "kousakis": "kou sakis", "koushika": "koushik a",
                                       "kousseri": "kousser i", "kremenchuh": "kremenchug",
                                       "kumarmangalam": "kumar mangalam", "kunstlerroman": "kunstler roman",
                                       "kushanas": "kusha nas", "laakman": "layman", "ladymen": "ladyboy",
                                       "lakhsman": "lakhs man", "lamenectomy": "lamnectomy",
                                       "languagetool": "language tool",
                                       "laowhy86": "foreigners who do not respect china",
                                       "latinamericans": "latin americans", "lawforcement": "law forcement",
                                       "lawyergirlfriend": "lawyer girl friend", "le'ts": "let us",
                                       "leewayhertz": "blockchain company", "legumnous": "leguminous",
                                       "lensmaker": "lens maker", "let's": "let us", "lethocerus": "lethocerus",
                                       "lexigographers": "lexicographers", "lifeaffect": "life affect",
                                       "liferature": "literature", "lifestly": "lifestyle", "lifestylye": "lifestyle",
                                       "lifeute": "life ute", "light'em": "light them",
                                       "likebjaish": "like bjaish", "likelogo": "like logo",
                                       "likelovequotes": "like lovequotes", "likemail": "like mail",
                                       "likemy": "like my", "like⬇": "like", "lingayatism": "lingayat",
                                       "lionese": "lioness", "lithromantics": "lith romantics",
                                       "lllustrate": "illustrate",
                                       "lndustrial": "industrial", "loadingtimes": "loading times",
                                       "lock'em": "lock them", "lock'um": "lock them", "lonewolfs": "lone wolfs",
                                       "look'em": "look them", "loundly": "loudly", "louspeaker": "loudspeaker",
                                       "love'em": "love them", "lovejihad": "love jihad",
                                       "lovestep": "love step", "loy machedeo": "person",
                                       "loy machedo": " motivational speaker ", "lucideus": "lucidum",
                                       "ludiculous": "ridiculous", "luscinus": "luscious", "lustfly": "lustful",
                                       "lustorus": "lustrous", "luxemgourg": "luxembourg", "ma'am": "madam",
                                       "macapugay": "macaulay", "machineworld": "machine world", "magetta": "maretta",
                                       "magibabble": "magi babble", "mailwoman": "mail woman",
                                       "maimonedes": "maimonides", "mainstreamly": "mainstream",
                                       "makedonian": "macedonian", "makeschool": "make school",
                                       "makeshifter": "make shifter",
                                       "makeup411": "makeup 411", "mamgement": "management",
                                       "manaagerial": "managerial", "managemebt": "management",
                                       "managemenet": "management", "managemental": "manage mental",
                                       "managementskills": "management skills", "managersworking": "managers working",
                                       "managewp": "managed", "manajement": "management",
                                       "manamement": "management", "manaufacturing": "manufacturing",
                                       "mandalikalu": "mandalika lu", "mandateing": "man dateing",
                                       "mandatkry": "mandatory", "mandingan": "mandingan",
                                       "mandrillaris": "mandrill aris", "maneuever": "maneuver",
                                       "mangalasutra": "mangalsutra", "mangalik": "manglik",
                                       "mangekyu": "mange kyu", "mangolian": "mongolian", "mangoliod": "mongoloid",
                                       "mangonada": "mango nada", "mangopay": "mango pay", "manholding": "man holding",
                                       "manhuas": "mahuas", "manies": "many", "manipative": "mancipative",
                                       "manipulant": "manipulate",
                                       "manipullate": "manipulate", "maniquins": "mani quins", "manjha": "mania",
                                       "mankirt": "mankind", "mankrit": "mank rit", "manlet": "man let",
                                       "manniya": "mania", "mannualy": "annual", "manorialism": "manorial ism",
                                       "manpads": "man pads",
                                       "manrega": "manresa", "mansatory": "mandatory", "manslamming": "mans lamming",
                                       "mansoon": "man soon", "manspread": "man spread",
                                       "manspreading": "man spreading", "manstruate": "menstruate",
                                       "mansturbate": "masturbate", "manterrupting": "interrupting",
                                       "manthras": "mantras",
                                       "manufacctured": "manufactured", "manufacturig": "manufacturing",
                                       "manufctures": "manufactures", "manufraturer": "manufacturer",
                                       "manufraturing": "manufacturing", "manufucturing": "manufacturing",
                                       "manupalation": "manipulation", "manupulative": "manipulative",
                                       "manuscriptology": "manuscript ology", "manvantar": "manvantara",
                                       "manwould": "man would", "manwues": "manages", "many4": "many",
                                       "manyare": "many are", "manychat": "many chat",
                                       "manycountries": "many countries", "manygovernment": "many government",
                                       "manyness": "many ness", "manyother": "many other", "maralago": "mar-a-lago",
                                       "maratis": "maratism", "marcusean": "marcuse an", "marimanga": "mari manga",
                                       "marionettist": "marionettes", "marlstone": "marls tone",
                                       "marrakush": "marrakesh", "massahusetts": "massachusetts",
                                       "masskiller": "mass killer", "mastercolonel": "master colonel",
                                       "mastubrate": "masturbate",
                                       "mastuburate": "masturbate", "mathusla": "mathusala", "mauritious": "mauritius",
                                       "maurititus": "mauritius", "mausturbate": "masturbate", "mayn't": "may not",
                                       "mcgreggor": "mcgregor", "measument": "measurement",
                                       "meausrements": "measurements", "medicalperson": "medical person",
                                       "meesaya": "mee saya", "megabuses": "mega buses", "megatapirus": "mega tapirus",
                                       "mellophones": "mellophone s", "memoney": "money", "menberships": "memberships",
                                       "mendalin": "mend alin", "mendatory": "mandatory", "menditory": "mandatory",
                                       "menedatory": "mandatory",
                                       "menifest": "manifest", "menifesto": "manifesto", "menigioma": "meningioma",
                                       "meninist": "male chauvinism", "meniss": "menise", "menmium": "medium",
                                       "mensrooms": "mens rooms", "menstrat": "menstruate", "menstrated": "menstruated",
                                       "menstraution": "menstruation",
                                       "menstruateion": "menstruation", "menstrute": "menstruate",
                                       "menstrution": "menstruation", "menstruual": "menstrual",
                                       "menstuating": "menstruating", "mensurational": "mensuration al",
                                       "mentalitiy": "mentality", "mentalized": "metalized",
                                       "mentenance": "maintenance", "mentionong": "mentioning",
                                       "mentiri": "entire", "mercadopago": "mercado pago", "meritious": "meritorious",
                                       "merrigo": "merligo", "messenget": "messenger",
                                       "metacompartment": "meta compartment", "metaphosphates": "meta phosphates",
                                       "methedone": "methadone", "mevius": "medius", "michrophone": "microphone",
                                       "microaggression": "micro aggression", "microapologize": "micro apologize",
                                       "microneedling": "micro needling", "microservices": "micro services",
                                       "microskills": "micros kills", "middleperson": "middle person",
                                       "mifeprostone": "mifepristone", "might've": "might have",
                                       "mightn't": "might not", "mightn't've": "might not have",
                                       "milkwithout": "milk without", "milo yianopolous": "a british polemicist",
                                       "minangkabaus": "minangkabau s", "minderheid": "minder worse",
                                       "misandrous": "misandry", "mismanagements": "mis managements",
                                       "missuses": "miss uses", "mistworlds": "mist worlds", "moduslink": "modus link",
                                       "mogolia": "mongolia",
                                       "momsays": "moms ays", "momuments": "monuments", "monegasques": "monegasque s",
                                       "monetation": "moderation", "moneycard": "money card", "moneycash": "money cash",
                                       "moneydriven": "money driven", "moneyfront": "money front", "moneyof": "mony of",
                                       "mongodump": "mongo dump",
                                       "mongoimport": "mongo import", "mongorestore": "mongo restore",
                                       "monoceious": "monoecious", "monogmous": "monogamous",
                                       "monosexuality": "mono sexuality", "montheistic": "nontheistic",
                                       "montogo": "montego", "moongot": "moong ot", "moraled": "morale",
                                       "motorious": "notorious",
                                       "mountaneous": "mountainous", "mouseflow": "mouse flow",
                                       "mousestats": "mouses tats", "moushmee": "mousmee", "mousquitoes": "mosquitoes",
                                       "moussolini": "mussolini", "moustachesomething": "moustache something",
                                       "moustachess": "moustaches", "movielush": "movie lush", "moviments": "movements",
                                       "msitake": "mistake", "muccus": "mucous", "muchdeveloped": "much developed",
                                       "muscleblaze": "muscle blaze", "muscluar": "muscular", "muscualr": "muscular",
                                       "musevi": "the independence of mexico", "musharrf": "musharraf",
                                       "mushlims": "muslims", "musickers": "musick ers",
                                       "musigma": "mu sigma", "musilim": "muslim", "musilms": "muslims",
                                       "musims": "muslims", "musino": "musion", "muslimare": "muslim are",
                                       "muslimophobe": "muslim phobic", "muslimophobia": "muslim phobia",
                                       "muslisms": "muslims", "muslium": "muslim",
                                       "mussraff": "muss raff", "must've": "must have", "mustabating": "must abating",
                                       "mustang1": "mustangs", "mustansiriyah": "mustansiriya h",
                                       "mustectomy": "mastectomy", "mustn't": "must not", "mustn't've": "must not have",
                                       "musturbation": "masturbation", "musuclar": "muscular",
                                       "mutiliating": "mutilating", "mutilitated": "mutilated",
                                       "muzaffarbad": "muzaffarabad", "mylogenous": "myogenous",
                                       "mystakenly": "mistakenly", "mythomaniac": "mythomania",
                                       "nagamandala": "naga mandala", "nagetive": "native", "naggots": "faggots",
                                       "namaj": "namaz",
                                       "namit bathla": "content writer", "narcsissist": "narcissist",
                                       "narracist": "nar racist", "narsistic": "narcistic",
                                       "nationalpost": "national post", "nauget": "naught", "nauseatic": "nausea tic",
                                       "nautlius": "nautilus", "nearbuy": "nearby", "neculeus": "nucleus",
                                       "needn't": "need not", "needn't've": "need not have", "negetively": "negatively",
                                       "negosiation": "negotiation", "negotiatiations": "negotiations",
                                       "negotiatior": "negotiation", "negotiotions": "negotiations",
                                       "neigous": "nervous", "nennus": "genius", "nesraway": "nearaway",
                                       "nestaway": "nest away", "neucleus": "nucleus", "neulife": "neu life",
                                       "neurosemantics": "neuro semantics", "neurosexist": "neuro sexist",
                                       "neuseous": "nauseous", "neverhteless": "nevertheless",
                                       "neverunlearned": "never unlearned", "newkiller": "new killer",
                                       "newscommando": "news commando",
                                       "nibirus": "nibiru", "nicholus": "nicholas", "nicus": "nidus",
                                       "niggor": "black hip-hop and electronic artist", "nitrogenious": "nitrogenous",
                                       "nobushi": "no bushi", "nomanclature": "nomenclature",
                                       "nonabandonment": "non abandonment", "nonchristians": "non christians",
                                       "noneconomically": "non economically",
                                       "nonelectrolyte": "non electrolyte", "nonexsistence": "nonexistence",
                                       "nonfermented": "non fermented", "nonhindus": "non hindus",
                                       "nonskilled": "non skilled", "nonspontaneous": "non spontaneous",
                                       "nonviscuous": "nonviscous", "noonein": "noo nein", "northcap": "north cap",
                                       "northestern": "northwestern",
                                       "nosebone": "nose bone", "nothinking": "thinking", "notmusing": "not musing",
                                       "nuchakus": "nunchakus", "nuclus": "nucleus", "nullifed": "nullified",
                                       "nuslims": "muslims", "nutrament": "nutriment", "nutriteous": "nutritious",
                                       "o'bamacare": "obamacare",
                                       "o'clock": "of the clock", "obfuscaton": "obfuscation",
                                       "objectmake": "object make", "obnxious": "obnoxious", "obumblers": "bumblers",
                                       "ofkulbhushan": "of kulbhushan", "ofvodafone": "of vodafone",
                                       "ointmentsointments": "ointments ointments",
                                       "oligodendraglioma": "oligodendroglioma", "olnhausen": "olshausen",
                                       "omnisexuality": "omni sexuality", "one300": "one 300", "oneblade": "one blade",
                                       "onecoin": "one coin", "onepiecedeals": "onepiece deals",
                                       "oneplus": "chinese smartphone manufacturer", "onsocial": "on social",
                                       "oogonial": "oogonia l", "orangetheory": "orange theory",
                                       "otherstates": "others tates",
                                       "ou're": "you are", "oughtn't": "ought not", "oughtn't've": "ought not have",
                                       "ousmania": "ous mania", "outfocus": "out focus", "outlook365": "outlook 365",
                                       "outonomous": "autonomous", "overcold": "over cold",
                                       "overcomeanxieties": "overcome anxieties", "overfeel": "over feel",
                                       "overjustification": "over justification", "overproud": "over proud",
                                       "overvcome": "overcome", "oxandrolonesteroid": "oxandrolone steroid",
                                       "ozonedepletion": "ozone depletion", "p***": "porn", "p****": "pussy",
                                       "p*rn": "porn", "p*ssy": "pussy", "p0 rnstars": "pornstars",
                                       "padmanabhanagar": "padmanabhan agar", "painterman": "painter man",
                                       "pakistainies": "pakistanis", "pakistanisbeautiful": "pakistanis beautiful",
                                       "pakustan": "pakistan", "palsmodium": "plasmodium", "palusami": "palus ami",
                                       "pangolinminer": "pangolin miner", "pangoro": "cantankerous pokemon",
                                       "panishments": "punishments",
                                       "panromantic": "pan romantic", "pansexuals": "pansexual",
                                       "pantherous": "panther ous", "papermoney": "paper money",
                                       "paracommando": "para commando", "parasuramans": "parasuram ans",
                                       "parilment": "parchment", "parkistan": "pakistan", "parkistinian": "pakistani",
                                       "parlamentarians": "parliamentarians",
                                       "parlamentary": "parliamentary", "parlementarian": "parlement arian",
                                       "parlimentry": "parliamentary", "parmenent": "permanent",
                                       "parmently": "patiently", "parralels": "parallels", "pasmanda": "pas manda",
                                       "patitioned": "petitioned", "pay'um": "pay them", "peactime": "peacetime",
                                       "pedogogical": "pedological", "pegusus": "pegasus",
                                       "peloponesian": "peloponnesian", "pentanone": "penta none",
                                       "peoplekind": "people kind", "peoplelike": "people like",
                                       "perfoemance": "performance", "performancelearning": "performance learning",
                                       "performancetesting": "performance testing", "performancies": "performances",
                                       "perliament": "parliament", "permanentjobs": "permanent jobs",
                                       "permanmently": "permanently", "permenganate": "permanganate",
                                       "persoenlich": "person lich", "personaltiles": "personal titles",
                                       "personifaction": "personification", "personlich": "person lich",
                                       "personslized": "personalized", "persulphates": "per sulphates",
                                       "petrostates": "petro states", "pettypotus": "petty potus", "ph.d": "phd",
                                       "pharisaistic": "pharisaism", "phenonenon": "phenomenon",
                                       "pheramones": "pheromones", "philanderous": "philander ous",
                                       "phinneus": "phineus", "phlegmonous": "phlegmon ous", "phnomenon": "phenomenon",
                                       "phonecases": "phone cases", "photoacoustics": "photo acoustics",
                                       "photofeeler": "photo feeler", "phusicist": "physicist",
                                       "phythagoras": "pythagoras", "pick'em": "pick them",
                                       "picosulphate": "pico sulphate", "pictones": "pict ones", "pingo5": "pingo",
                                       "pizza gate": "debunked conspiracy theory",
                                       "placdment": "placement", "plaement": "placement", "plagetum": "plage tum",
                                       "platfrom": "platform", "playerunknown": "player unknown",
                                       "playmagnus": "play magnus", "pleasegive": "please give",
                                       "pliosaurus": "pliosaur us", "plus5": "plus", "plustwo": "plus two",
                                       "poisenious": "poisonous", "poisiones": "poisons", "pokestops": "pokes tops",
                                       "polishment": "pol ishment", "politicak": "political", "polonious": "polonius",
                                       "polyagamous": "polygamous", "polygomists": "polygamists",
                                       "polygony": "poly gony", "polyhouse": "polytunnel",
                                       "polyhouses": "polytunnel", "poneglyphs": "pone glyphs",
                                       "pornosexuality": "porno sexuality", "posinous": "rosinous",
                                       "posionus": "poisons", "postincrement": "post increment",
                                       "postmanare": "postman are", "powerballsusa": "powerballs usa",
                                       "prechinese": "pre chinese", "prefomance": "performance",
                                       "pregnantwomen": "pregnant women", "preincrement": "pre increment",
                                       "prejusticed": "prejudiced", "prelife": "pre life", "prelimenary": "preliminary",
                                       "prendisone": "prednisone", "prentious": "pretentious",
                                       "presumptuousnes": "presumptuousness", "pretenious": "pretentious",
                                       "pretex": "pretext",
                                       "pretextt": "pre text", "preussure": "pressure", "previius": "previous",
                                       "priebuss": "prie buss", "primarty": "primary", "probationees": "probationers",
                                       "procument": "procumbent", "prodimently": "prominently",
                                       "productionsupport": "production support",
                                       "productmanagement": "product management",
                                       "productsexamples": "products examples", "programmebecause": "programme because",
                                       "programmingassignments": "programming assignments", "promenient": "provenient",
                                       "promuslim": "pro muslim", "propelment": "propel ment",
                                       "prospeorus": "prosperous", "prosporously": "prosperously",
                                       "protopeterous": "protopterous", "provocates": "provokes",
                                       "prussophile": "russophile", "pseudomeningocele": "pseudo meningocele",
                                       "psiphone": "psi phone", "publious": "publius", "pulmanery": "pulmonary",
                                       "pulphus": "pulpous", "puniahment": "punishment", "purusharthas": "purushartha",
                                       "purushottampur": "purushottam pur", "pushbullet": "push bullet",
                                       "pushkaram": "pushkara m", "puspak": "pu spak", "pussboy": "puss boy",
                                       "pyromantic": "pyro mantic", "qmas": "quality migrant admission scheme",
                                       "qoura": "quora", "qualifeid": "qualified", "queffing": "queefing",
                                       "qurush": "qu rush", "r - apist": "rapist",
                                       "ra - apist": "rapist", "ra apist": "rapist",
                                       "racistcomments": "racist comments", "racistly": "racist", "raddus": "radius",
                                       "radijus": "radius", "rahmanland": "rahman land", "rajsthan": "rajasthan",
                                       "rajsthanis": "rajasthanis", "ramanunjan": "ramanujan",
                                       "rammandir": "ram mandir", "rammayana": "ramayana", "rangeman": "range man",
                                       "rangermc": "car", "rankholders": "rank holders", "rapefilms": "rape films",
                                       "rapiest": "rapist", "rasayanam": "rasayan am", "ratkill": "rat kill",
                                       "realbonus": "real bonus",
                                       "reallysemites": "really semites", "realtimepolitics": "realtime politics",
                                       "recmommend": "recommend", "recommendor": "recommender",
                                       "recommening": "recommending", "recordermans": "recorder mans",
                                       "recrecommend": "rec recommend", "recruitment2017": "recruitment 2017",
                                       "recrument": "recrement", "recuirement": "requirement",
                                       "recurtment": "recurrent", "recusion": "recushion", "recussed": "recursed",
                                       "redicules": "ridiculous", "redius": "radius", "redmi": "xiaomi mobile",
                                       "refernece": "reference", "refundment": "refund ment", "regements": "regiments",
                                       "regilious": "religious",
                                       "reglamented": "reg lamented", "regognitions": "recognitions",
                                       "regognized": "recognized", "reimbrusement": "reimbursement",
                                       "reinfectus": "reinfect", "rejectes": "rejected", "reliegious": "religious",
                                       "religiouslike": "religious like", "remainers": "remainder",
                                       "remaninder": "remainder",
                                       "remenant": "remnant", "remmaning": "remaining", "rendementry": "rendement ry",
                                       "repariments": "departments", "replecement": "replacement",
                                       "representment": "rep resentment", "repugicans": "republicans",
                                       "repurcussion": "repercussion", "repyament": "repayment",
                                       "requairment": "requirement",
                                       "requriment": "requirement", "requriments": "requirements",
                                       "requsite": "requisite", "requsitioned": "requisitioned",
                                       "rescouses": "responses", "resigment": "resignment", "reskilled": "skilled",
                                       "resustence": "resistence", "retairment": "retainment",
                                       "reteirement": "retirement",
                                       "retrocausality": "retro causality", "retrovisus": "retrovirus",
                                       "returement": "retirement", "reusibility": "reusability", "reusult": "result",
                                       "reverificaton": "reverification", "reveuse": "reve use",
                                       "rewardingways": "rewarding ways", "rheusus": "rhesus", "richfeel": "rich feel",
                                       "richmencupid": "rich men dating website", "ridicjlously": "ridiculously",
                                       "rigetti": "ligetti", "righteouness": "righteousness", "rigrously": "rigorously",
                                       "rimworld": "rim world", "ringostat": "ringo stat",
                                       "rivigo": "technology-enabled logistics company", "roadgods": "road gods",
                                       "rohmbus": "rhombus",
                                       "rolloverbox": "rollover box", "romancetale": "romance tale",
                                       "romanium": "romanum", "romantize": "romanize", "rombous": "bombous",
                                       "routez": "route", "rovman": "roman", "roxycodone": "r oxycodone",
                                       "royago": "royal", "rpatah - tan eng hwan": "silsilah",
                                       "rseearch": "research", "rstman": "rotman", "rubustness": "robustness",
                                       "rumenova": "rumen ova", "run'em": "run them", "russiagate": "russia gate",
                                       "russions": "russians", "russophobic": "russophobiac",
                                       "russosphere": "russia sphere of influence", "rustichello": "rustic hello",
                                       "rustyrose": "rusty rose", "s**": "shi", "s***": "shit", "saadus": "status",
                                       "sadhgurus": "sadh gurus", "saffronize": "india, politics, derogatory",
                                       "saffronized": "india, politics, derogatory", "saggittarius": "sagittarius",
                                       "sagittarious": "sagittarius", "salemmango": "salem mango",
                                       "salesmanago": "salesman ago", "sallatious": "fallacious", "samousa": "samosa",
                                       "sapiosexual": "sexually attracted to intelligence",
                                       "sapiosexuals": "sapiosexual", "sarumans": "sarum ans", "sasabone": "sasa bone",
                                       "satannus": "sat annus", "sauskes": "causes", "savegely": "savagely",
                                       "savethechildren": "save thechildren", "savonious": "sanious",
                                       "savvius": "savvies", "saydaw": "say daw", "saynthesize": "synthesize",
                                       "sayying": "saying", "sb91": "senate bill", "scomplishments": "accomplishments",
                                       "scuduse": "scud use", "searious": "serious",
                                       "securedlife": "secured life", "secutus": "sects", "sedataious": "seditious",
                                       "seeies": "series", "seekingmillionaire": "seeking millionaire", "see‬": "see",
                                       "selfknowledge": "self knowledge", "selfpayment": "self payment",
                                       "semisexual": "semi sexual", "send'em": "send them",
                                       "senousa": "venous", "serieusly": "seriously", "seriousity": "seriosity",
                                       "settelemen": "settlement", "settlementtake": "settlement take",
                                       "setya novanto": "a former indonesian politician", "sevenfriday": "seven friday",
                                       "sevenpointed": "seven pointed", "seventysomething": "seventy something",
                                       "severiity": "severity",
                                       "seviceman": "serviceman", "sexeverytime": "sex everytime",
                                       "sexgods": "sex gods", "sexitest": "sexiest", "sexlike": "sex like",
                                       "sexmates": "sex mates", "sexond": "second", "sexpat": "sex tourism",
                                       "sexsurrogates": "sex surrogates", "sextactic": "sex tactic",
                                       "sexualises": "sexualise", "sexualityism": "sexuality ism",
                                       "sexuallly": "sexually", "sexualslutty": "sexual slutty", "sexuly": "sexily",
                                       "sexxual": "sexual", "sexyjobs": "sexy jobs", "sh**": "shit", "sh*tty": "shit",
                                       "sha'n't": "shall not",
                                       "shan't": "shall not", "shan't've": "shall not have", "she''l": "she will",
                                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                                       "she'll've": "she will have", "she's": "she is", "shitpeople": "shit people",
                                       "shoot'um": "shoot them",
                                       "shoudn't": "should not", "should've": "should have", "shouldn't": "should not",
                                       "shouldn't've": "should not have", "shouldntake": "shouldnt take",
                                       "showh": "show", "shrusti": "shruti", "shubman": "subman",
                                       "shuoldnt": "shouldnt", "sigorn": "son of styr",
                                       "silghtest": "slightest", "siliconindia": "silicon india",
                                       "silumtaneously": "simultaneously", "simantaneously": "simultaneously",
                                       "simentaneously": "simultaneously", "simoultaneously": "simultaneously",
                                       "simultameously": "simultaneously", "simultenously": "simultaneously",
                                       "simultqneously": "simultaneously", "simutaneusly": "simultaneously",
                                       "simutanously": "simultaneously", "sinophone": "sinophobe",
                                       "sissyphus": "sisyphus", "sisterzoned": "sister zoned",
                                       "sjws": "social justice warrior", "skillclasses": "skill classes",
                                       "skillport": "army e-learning program", "skillselect": "skills elect",
                                       "skyhold": "sky hold", "slavetrade": "slave trade",
                                       "sllaybus": "syllabus", "sllyabus": "syllabus", "sllybus": "syllabus",
                                       "sloatman": "sloat man", "slogun": "slogan", "sloppers": "slippers",
                                       "slovenised": "slovenia", "smarthpone": "smartphone",
                                       "smartworld": "sm artworld", "smelllike": "smell like",
                                       "snapragon": "snapdragon", "snazzyway": "snazzy way",
                                       "sneakerlike": "sneaker like", "snuus": "snugs", "so's": "so as",
                                       "so've": "so have", "sociomantic": "sciomantic", "softskill": "softs kill",
                                       "softthinks": "soft thinks", "soldiders": "soldiers",
                                       "som'thin": "something", "someonewith": "some onewith",
                                       "southafricans": "south africans", "southeners": "southerners",
                                       "southerntelescope": "southern telescope", "southindia": "south india",
                                       "soyboys": "cuck men lacking masculine characteristics",
                                       "spacewithout": "space without", "spause": "spouse", "speakingly": "speaking",
                                       "specrum": "spectrum", "spectrastone": "spectra stone",
                                       "spectulated": "speculated", "speicial": "special",
                                       "sponataneously": "spontaneously", "sponteneously": "spontaneously",
                                       "spoolman": "spool man", "sportsusa": "sports usa", "sppliment": "supplement",
                                       "spymania": "spy mania",
                                       "sqamous": "squamous", "sreeman": "freeman", "sshouldn't": "should not",
                                       "st*up*id": "stupid", "staionery": "stationery",
                                       "stargold": "a hindi movie channel", "starseeders": "star seeders",
                                       "startfragment": "start fragment", "stategovt": "state govt",
                                       "statemment": "statement",
                                       "statusguru": "status guru", "stegosauri": "stegosaurus",
                                       "stelouse": "ste louse", "steymann": "stedmann", "stillshots": "stills hots",
                                       "stillsuits": "still suits", "stocklogos": "stock logos", "stomuch": "stomach",
                                       "stonehart": "stone hart", "stonemen": "stone men",
                                       "stonepelters": "stone pelters", "stopings": "stoping", "stoppef": "stopped",
                                       "stoppingexercises": "stopping exercises", "stopsigns": "stop signs",
                                       "stopsits": "stop sits", "straitstimes": "straits times",
                                       "straussianism": "straussian ism", "stupidedt": "stupidest", "stusy": "study",
                                       "subcautaneous": "subcutaneous", "subcentimeter": "sub centimeter",
                                       "subconcussive": "sub concussive", "subconsciousnesses": "sub consciousnesses",
                                       "subligamentous": "sub ligamentous", "subramaniyan": "subramani yan",
                                       "suchvstupid": "such stupid", "suconciously": "unconciously",
                                       "sulamaniya": "sulamani ya", "sulmann": "suilmann",
                                       "sulprus": "surplus", "sumaterans": "sumatrans", "sunnyleone": "sunny leone",
                                       "sunstop": "sun stop", "suparwoman": "superwoman",
                                       "supermaneuverability": "super maneuverability",
                                       "supermaneuverable": "super maneuverable", "superowoman": "superwoman",
                                       "superplus": "super plus", "supersynchronous": "super synchronous",
                                       "supertournaments": "super tournaments", "supplemantary": "supplementary",
                                       "supplemenary": "supplementary", "supplementplatform": "supplement platform",
                                       "supplymentary": "supply mentary", "supplymentry": "supplementary",
                                       "surgetank": "surge tank", "susanoomon": "susanoo mon",
                                       "susbtraction": "substraction", "susgaon": "surgeon",
                                       "sushena": "saphena", "suspectance": "suspect ance", "suspeect": "suspect",
                                       "suspenive": "suspensive", "suspicius": "suspicious", "sussessful": "successful",
                                       "sussia": "ancient jewish village", "sustainabke": "sustainable",
                                       "sustinet": "sustinent", "susubsoil": "su subsoil",
                                       "swayable": "sway able", "swissgolden": "swiss golden", "syallbus": "syllabus",
                                       "syallubus": "syllabus", "sychronous": "synchronous", "sylaabus": "syllabus",
                                       "sylabbus": "syllabus", "syllaybus": "syllabus", "syncway": "sync way",
                                       "tagushi": "tagus hi",
                                       "taharrush": "tahar rush", "take'em": "take them", "takeove": "takeover",
                                       "takeoverr": "takeover", "takeoverrs": "takeovers", "takesuch": "take such",
                                       "takingoff": "taking off", "talecome": "tale come", "tamanaa": "tamanac",
                                       "tarumanagara": "taruma nagara",
                                       "taskus": "", "tastaman": "rastaman", "teamtreehouse": "team treehouse",
                                       "techinacal": "technical", "techmakers": "tech makers",
                                       "technoindia": "techno india", "teethbrush": "teeth brush",
                                       "telegraphindia": "telegraph india", "tell'em": "tell them",
                                       "telloway": "tello way",
                                       "tennesseus": "tennessee", "terimanals": "terminals", "terroristan": "terrorist",
                                       "testostersone": "testosterone",
                                       "testoultra": "male sexual enhancement supplement",
                                       "teststerone": "testosterone", "tetherusd": "tethered",
                                       "tetramerous": "tetramer ous", "tetraosulphate": "tetrao sulphate",
                                       "tfws": "tuition fee waiver",
                                       "tgethr": "together", "thaedus": "thaddus", "thammana": "tamannaah bhatia",
                                       "that''s": "that is", "that'd": "that would", "that'd've": "that would have",
                                       "that'll": "that will", "that's": "that is", "thateasily": "that easily",
                                       "thausand": "thousand",
                                       "theeventchronicle": "the event chronicle",
                                       "theglobeandmail": "the globe and mail", "theguardian": "the guardian",
                                       "ther'es": "there is", "there'd": "there would",
                                       "there'd've": "there would have", "there's": "there is",
                                       "thereanyone": "there anyone", "theuseof": "thereof", "they'd": "they would",
                                       "they'd've": "they would have", "they'er": "they are", "they'l": "they will",
                                       "they'll": "they will", "they'll've": "they will have", "they'lll": "they will",
                                       "they're": "they are", "they've": "they have", "they_didn't": "they did not",
                                       "theyr'e": "they are",
                                       "theyreally": "they really", "theyv'e": "they have", "thiests": "atheists",
                                       "thinkinh": "thinking", "thinkstrategic": "think strategic",
                                       "thinksurvey": "think survey", "this's": "this is",
                                       "thisaustralian": "this australian", "thrir": "their", "tiannanmen": "tiananmen",
                                       "tiltbrush": "tilt brush", "timefram": "timeframe", "timejobs": "time jobs",
                                       "timeloop": "time loop", "timesence": "times ence", "timesjobs": "times jobs",
                                       "timesnow": "24-hour english news channel in india", "timesspark": "times spark",
                                       "timesup": "times up", "timetabe": "timetable",
                                       "timetraveling": "timet raveling", "timetraveller": "time traveller",
                                       "timetravelling": "timet ravelling", "timewaste": "time waste",
                                       "to've": "to have", "toastsexual": "toast sexual", "tobagoans": "tobago ans",
                                       "togofogo": "togo fogo", "tonogenesis": "tone", "toostupid": "too stupid",
                                       "toponymous": "top onymous", "tormentous": "torment ous",
                                       "torvosaurus": "torosaurus", "totalinvestment": "total investment",
                                       "touchtime": "touch time", "towayrds": "towards",
                                       "tradecommander": "trade commander", "tradeplus": "trade plus",
                                       "trageting": "targeting", "trampaphobia": "trump aphobia",
                                       "tranfusions": "transfusions", "transexualism": "transsexualism",
                                       "transgenus": "trans genus", "transness": "trans gender",
                                       "transtrenders": "incredibly disrespectful to real transgender people",
                                       "trausted": "trusted", "treatens": "threatens", "treatmenent": "treatment",
                                       "treatmentshelp": "treatments help", "treetment": "treatment",
                                       "tremendeous": "tremendous", "triangleright": "triangle right",
                                       "tricompartmental": "tri compartmental", "tridentinus": "mushroom",
                                       "trigonomatry": "trigonometry", "trillonere": "trillones",
                                       "trimegistus": "trismegistus", "trimp": "trump",
                                       "triphosphorus": "tri phosphorus", "trueoutside": "true outside",
                                       "trump": "trump", "trumpcare": "trump health care system",
                                       "trumpdating": "trump dating", "trumpdoesn'tcare": "trump does not care",
                                       "trumpdon'tcareact": "trump do not care act", "trumpers": "trumpster",
                                       "trumpervotes": "trumper votes",
                                       "trumpian": "viewpoints of president donald trump",
                                       "trumpidin'tcare": "trump did not care",
                                       "trumpism": "philosophy and politics espoused by donald trump",
                                       "trumpists": "admirer of donald trump", "trumpites": "trump supporters",
                                       "trumplies": "trump lies", "trumpology": "trump ology",
                                       "trumpster": "trumpeters", "trumpsters": "trump supporters",
                                       "trustclix": "trust clix", "trustkit": "trust kit", "trustless": "t rustless",
                                       "trustworhty": "trustworthy",
                                       "trustworhy": "trustworthy", "tusaki": "tu saki", "tusami": "tu sami",
                                       "tusts": "trusts", "twiceusing": "twice using", "twinflame": "twin flame",
                                       "tyrannously": "tyrannous", "u'r": "you are", "u.k.": "uk", "u.s.": "usa",
                                       "u.s.a": "usa", "u.s.a.": "usa", "u.s.p": "", "ucsandiego": "uc sandiego",
                                       "ugggggggllly": "ugly", "uimovement": "ui movement", "umumoney": "umu money",
                                       "unacademy": "educational technology company", "unamendable": "un amendable",
                                       "unanonymously": "un anonymously",
                                       "unblacklisted": "un blacklisted", "uncosious": "uncopious",
                                       "uncouncious": "unconscious", "underdevelopement": "under developement",
                                       "undergraduation": "under graduation", "understamding": "understanding",
                                       "understandment": "understand ment", "undertale": "video game",
                                       "underthinking": "under thinking", "undervelopment": "undevelopment",
                                       "underworldly": "under worldly", "unergonomic": "un ergonomic",
                                       "unforgottable": "unforgettable", "unglaus": "ung laus",
                                       "uninstrusive": "unintrusive", "unitedstatesian": "united states",
                                       "unkilled": "un killed", "unmoraled": "unmoral",
                                       "unpermissioned": "unper missioned", "unrightly": "un rightly",
                                       "unwanted72": "unwanted 72", "upcomedians": "up comedians", "upwork": "up work",
                                       "urotone": "protone", "usa''s": "usa", "usagovernment": "usa government",
                                       "use38": "use", "usebase": "use base", "usedtoo": "used too",
                                       "usedvin": "used vin",
                                       "usefl": "useful", "userbags": "user bags", "userflows": "user flows",
                                       "usertesting": "user testing", "useul": "useful", "ushually": "usually",
                                       "usict": "ussct", "uslme": "some", "uspset": "upset",
                                       "usucaption": "usu caption",
                                       "utilitas": "utilities", "utmterm": "utm term", "vairamuthus": "vairamuthu s",
                                       "vancouever": "vancouver", "vegetabale": "vegetable", "vegetablr": "vegetable",
                                       "vegetarean": "vegetarian", "vegetaries": "vegetables",
                                       "venetioned": "venetianed", "venus25": "venus",
                                       "vertigos": "vertigo s", "vetronus": "verrons", "vibhushant": "vibhushan t",
                                       "vicevice": "vice vice", "vinis": "vinys", "virituous": "virtuous",
                                       "virushka": "great relationships couple", "visahouse": "visa house",
                                       "vishnus": "vishnu", "vitilgo": "vitiligo",
                                       "vivipoarous": "viviparous", "vodafone2": "vodafones", "volime": "volume",
                                       "vote'em": "vote them", "votebanks": "vote banks", "w'ell": "we will",
                                       "wanket": "wanker", "wannaone": "wanna one", "wantrank": "want rank",
                                       "washingtontimes": "washington times",
                                       "washwoman": "wash woman", "wasn't": "was not", "watchtime": "watch time",
                                       "wattman": "watt man", "we''ll": "we will", "we'd": "we would",
                                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                                       "we're": "we are",
                                       "we'really": "we are really", "we've": "we have", "weatern": "western",
                                       "webdevelopement": "web developement", "webmusic": "web music",
                                       "wedgieman": "wedgie man", "wedugo": "wedge", "weenus": "elbow skin",
                                       "weightwithout": "weight without", "welcomemarriage": "welcome marriage",
                                       "welcomeromanian": "welcome romanian", "wellsfargoemail": "wellsfargo email",
                                       "wenzeslaus": "wenceslaus", "weren't": "were not", "wern't": "were not",
                                       "westsouth": "west south", "what sapp": "whatsapp", "what'll": "what will",
                                       "what'll've": "what will have", "what're": "what are",
                                       "what's": "what is", "what've": "what have", "whatasapp": "whatsapp",
                                       "whatcould": "what could", "whatcus": "what cause",
                                       "whateducation": "what education", "whatevidence": "what evidence",
                                       "whatmakes": "what makes", "whatshapp": "whatsapp", "whatsupp": "whatsapp",
                                       "whattsup": "whatsapp", "whatwould": "what would", "wheatestone": "wheatstone",
                                       "whemever": "whenever", "when's": "when is", "when've": "when have",
                                       "where burkhas": "wear burqas", "where'd": "where did", "where's": "where is",
                                       "where've": "where have",
                                       "whichcountry": "which country", "whichtreatment": "which treatment",
                                       "whilemany": "while many", "whitegirls": "white girls",
                                       "whiteheds": "whiteheads", "whitelash": "white lash",
                                       "whitesplaining": "white splaining", "whitetning": "whitening",
                                       "whitewalkers": "white walkers", "who''s": "who is",
                                       "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                                       "who've": "who have", "whoimplemented": "who implemented", "whwhat": "what",
                                       "why's": "why is", "why've": "why have", "whybis": "why is",
                                       "whyco-education": "why co-education",
                                       "wildstone": "wilds tone", "will've": "will have", "willhappen": "will happen",
                                       "williby": "will by", "willowmagic": "willow magic", "willseye": "will eye",
                                       "winor": "win", "withgoogle": "with google", "withgott": "without",
                                       "withoutcheck": "without check",
                                       "withoutregistered": "without registered", "withoutyou": "without you",
                                       "womansplained": "womans plained", "womansplaining": "wo mansplaining",
                                       "womenizer": "womanizer", "won't": "will not", "won't've": "will not have",
                                       "won'tdo": "will not do", "workaway": "work away", "workdone": "work done",
                                       "workfusion": "work fusion", "workouses": "workhouses",
                                       "workperson": "work person", "worldbusiness": "world business",
                                       "worldkillers": "world killers", "worldmax": "wholesaler of drum parts",
                                       "worldquant": "world quant", "worldrank": "world rank",
                                       "worldwideley": "worldwide ley", "worstplatform": "worst platform",
                                       "would'nt": "would not", "would't": "would not", "would've": "would have",
                                       "wouldd": "would", "wouldn't": "would not", "wouldn't've": "would not have",
                                       "wowmen": "women", "ww 1": " ww1 ", "ww 2": " ww2 ", "y'all": "you all",
                                       "y'all'd": "you all would", "y'all'd've": "you all would have",
                                       "y'all're": "you all are", "y'all've": "you all have", "y'know": "you know",
                                       "yaerold": "year old", "yahtzees": "yahtzee", "yeardold": "years old",
                                       "yegorovich": "yegorov ich", "yo'ure": "you are",
                                       "you''re": "you are", "you'all": "you all", "you'd": "you would",
                                       "you'd've": "you would have", "you'ld": "you would", "you'll": "you will",
                                       "you'll've": "you will have", "you're": "you are",
                                       "you'reonyourowncare": "you are on your own care", "you'res": "you are",
                                       "you'rethinking": "you are thinking", "you've": "you have",
                                       "you'very": "you are very", "youbecome": "you become", "youbever": "you bever",
                                       "your'e": "you are", "yousician": "musician", "yoyou": "you",
                                       "yuguslavia": "yugoslavia", "yumstone": "yum stone",
                                       "yutyrannus": "yu tyrannus", "yᴏᴜ": "you", "zamusu": "amuse",
                                       "zenfone": "zen fone", "zhuchengtyrannus": "zhucheng tyrannus",
                                       "zigolo": "gigolo", "zoneflex": "zone flex", "zymogenous": "zymogen ous",
                                       "ʙᴏᴛtoᴍ": "bottom", "ι": "i",
                                       "υ": "u", "в": "b", "м": "m", "н": "h", "т": "t", "ѕ": "s", "ᴀ": "a",
                                       "™": "trade mark", "∠bad": "bad", "」come": "come",
                                       "操你妈": "fuck your mother", "走go": "go", "😀": "stuck out tongue", "😂": "joy",
                                       "😉": "wink", }

        self.translate_dictionary = {
            '²': '2', '¹': '1', 'ĝ': 'g', 'œ': 'ae', 'ŝ': 's', 'ǧ': 'g', 'ɑ': 'ɑ',
            'ɒ': 'a', 'ɔ': 'c', 'ə': 'e', 'ɛ': 'e', 'ɡ': 'g', 'ɢ': 'g', 'ɪ': 'i',
            'ɴ': 'n', 'ʀ': 'r', 'ʏ': 'y', 'ʙ': 'b', 'ʜ': 'h', 'ʟ': 'l', 'ʰ': 'h',
            'ʳ': 'r', 'ʷ': 'w', 'ʸ': 'y', 'ˢ': '5', '͞': '-', '͟': '_', 'ͦ': 'o',
            'έ': 'e', 'ί': 'i', 'α': 'a', 'κ': 'k', 'χ': 'x', 'І': 'i', 'А': 'a',
            'Б': 'e', 'З': '#', 'И': 'n', 'У': 'y', 'Х': 'x', 'в': 'b',
            'к': 'k', 'м': 'm', 'н': 'h', 'ы': 'bi', 'ь': 'b', 'ё': 'e', 'љ': 'jb',
            'ғ': 'f', 'ү': 'y', 'Ԝ': 'w', 'հ': 'h', 'א': 'n', '௦': '0', '౦': 'o',
            '൦': 'o', '໐': 'o', 'Ꭵ': 'i', 'Ꭻ': 'j', 'Ꮷ': 'd', 'ᐨ': '-', 'ᐸ': '<',
            'ᑲ': 'b', 'ᑳ': 'b', 'ᗞ': 'd', 'ᴀ': 'a', 'ᴄ': 'c', 'ᴅ': 'n', 'ᴇ': 'e',
            'ᴊ': 'j', 'ᴋ': 'k', 'ᴍ': 'm', 'ᴏ': 'o', 'ᴑ': 'o', 'ᴘ': 'p', 'ᴛ': 't',
            'ᴜ': 'u', 'ᴠ': 'v', 'ᴡ': 'w', 'ᴵ': 'i', 'ᴷ': 'k', 'ᴺ': 'n', 'ᴼ': 'o',
            'ᵉ': 'e', 'ᵒ': 'o', 'ᵗ': 't', 'ᵘ': 'u', 'ẃ': 'w', 'ἀ': 'a', 'Ἀ': 'a',
            'Ἄ': 'a', 'ὶ': 'l', 'ὺ': 'u', '‒': '-', '₁': '1', '₃': '3', '₄': '4',
            'ℋ': 'h', '℠': 'sm', 'ℯ': 'e', 'ℴ': 'c', '╌': '--', 'ⲏ': 'h', 'ⲣ': 'p',
            '下': 'under', '不': 'Do not', '人': 'people', '伎': 'trick', '会': 'meeting',
            '作': 'Make', '你': 'you', '克': 'Gram', '关': 'turn off', '别': 'do not',
            '加': 'plus', '华': 'China', '卖': 'Sell', '去': 'go with', '哥': 'brother',
            '园': 'garden', '国': 'country', '圆': 'circle', '土': 'soil', '地': 'Ground',
            '坏': 'Bad', '外': 'outer', '大': 'Big', '失': 'Lost', '子': 'child', '小': 'small',
            '成': 'to make', '戦': 'War', '所': 'Place', '拿': 'take', '故': 'Therefore',
            '文': 'Text', '明': 'Bright', '是': 'Yes', '有': 'Have', '歌': 'song',
            '殊': 'special', '油': 'oil', '温': 'temperature', '特': 'special',
            '獄': 'prison', '的': 'of', '税': 'tax', '系': 'system', '群': 'group',
            '舞': 'dance', '英': 'English', '蔡': 'Cai', '议': 'Discussion', '谷': 'Valley',
            '豆': 'beans', '都': 'All', '钱': 'money', '降': 'drop', '障': 'barrier',
            '骗': 'cheat', '세': 'three', '안': 'within', '영': 'spirit', '요': 'Yo',
            'ͺ': '', 'Λ': 'L', 'Ξ': 'X', 'ά': 'a', 'ή': 'or', 'ι': 'j',
            'ξ': 'X', 'ς': 's', 'ψ': 't', 'ό': 'The', 'ύ': 'gt;', 'ώ': 'o',
            'ϖ': 'e.g.', 'Г': 'R', 'Д': 'D', 'Ж': 'F', 'Л': 'L', 'П': 'P',
            'Ф': 'F', 'Ш': 'Sh', 'б': 'b', 'п': 'P', 'ф': 'f', 'ц': 'c',
            'ч': 'no', 'ш': 'sh', 'щ': 'u', 'э': 'uh', 'ю': 'Yu', 'ї': 'her',
            'ћ': 'ht', 'Ձ': 'Winter', 'ա': 'a', 'դ': 'd', 'ե': 'e', 'ի': 's',
            'ձ': 'h', 'մ': 'm', 'յ': 'y', 'ն': 'h', 'ռ': 'r', 'ս': 'c',
            'ր': 'p', 'ւ': '³', 'ב': 'B', 'ד': 'D', 'ה': 'God', 'ו': 'and',
            'ט': 'ninth', 'י': 'J', 'ך': 'D', 'כ': 'about', 'ל': 'To', 'ם': 'From',
            'מ': 'M', 'ן': 'Estate', 'נ': 'N', 'ס': 'S.', 'ע': 'P', 'ף': 'Jeff',
            'פ': 'F', 'צ': 'C', 'ק': 'K.', 'ר': 'R.', 'ש': 'That', 'ת': 'A',
            'ء': 'Was', 'آ': 'Ah', 'أ': 'a', 'إ': 'a', 'ا': 'a', 'ة': 'e',
            'ت': 'T', 'ج': 'C', 'ح': 'H', 'خ': 'Huh', 'د': 'of the', 'ر': 'T',
            'ز': 'Z', 'س': 'Q', 'ش': 'Sh', 'ص': 's', 'ط': 'I', 'ع': 'AS', 'غ': 'G',
            'ف': 'F', 'ق': 'S', 'ك': 'K', 'ل': 'to', 'م': 'M', 'ن': 'N', 'ه': 'e',
            'و': 'And', 'ى': 'I', 'ي': 'Y', 'چ': 'What', 'ک': 'K', 'ی': 'Y',
            'क': 'A', 'म': 'M', 'र': 'And', 'ગ': 'C', 'જ': 'The same',
            'ત': 'I', 'ર': 'I', 'ஜ': 'SAD', 'ლ': 'L', 'ṑ': 'o', 'ἐ': 'e',
            'ἔ': 'Ë', 'ἡ': 'or', 'ἱ': 'ı', 'ἴ': 'i', 'ὀ': 'The', 'ὁ': 'The',
            'ὐ': 'ÿ', 'ὰ': 'a', 'ὲ': '.', 'ὸ': 'The', 'ύ': 'gt;', 'ᾶ': 'a',
            'ῆ': 'or', 'ῖ': 'ก', 'ῦ': 'I', 'う': 'U', 'さ': 'The', 'っ': 'What',
            'つ': 'One', 'な': 'The', 'よ': 'The', 'ら': 'Et al', 'エ': 'The',
            'ク': 'The', 'サ': 'The', 'シ': 'The', 'ジ': 'The', 'ス': 'The',
            'チ': 'The', 'ツ': 'The', 'ニ': 'D', 'ハ': 'Ha', 'マ': 'Ma',
            'リ': 'The', 'ル': 'Le', 'レ': 'Les', 'ロ': 'The', 'ン': 'The',
            '一': 'One', '与': 'versus', '且': 'And', '为': 'for', '买': 'buy',
            '了': 'Up', '些': 'some', '他': 'he', '以': 'Take', '们': 'They',
            '件': 'Items', '传': 'pass', '伦': 'Lun', '但': 'but', '信': 'letter',
            '候': 'Waiting', '偽': 'Pseudo', '全': 'all', '公': 'public', '其': 'its',
            '养': 'support', '冬': 'winter', '凸': 'Convex', '击': 'hit', '判': 'Judge',
            '到': 'To', '友': 'Friend', '可': 'can', '吗': 'What?', '和': 'with',
            '唯': 'only', '因': 'because', '圣': 'Holy', '在': 'in', '基': 'base',
            '堂': 'Hall', '复': 'complex', '多': 'many', '天': 'day',
            '好': 'it is good', '如': 'Such as', '婚': 'marriage', '孩': 'child',
            '宠': 'Pet', '寓': 'Apartment', '对': 'Correct', '屁': 'fart',
            '屈': 'Qu', '巨': 'huge', '己': 'already', '式': 'formula', '当': 'when',
            '彼': 'he', '徒': 'only', '得': 'Got', '怒': 'angry', '怪': 'strange',
            '恐': 'fear', '惧': 'fear', '想': 'miss you', '愤': 'anger', '我': 'I',
            '战': 'war', '批': 'Batch', '把': 'Put', '拉': 'Pull', '拷': 'Copy',
            '接': 'Connect', '操': 'Fuck', '收': 'Receive', '政': 'Politics',
            '教': 'teach', '斤': 'jin', '斯': 'S', '新': 'new', '时': 'Time',
            '普': 'general', '曾': 'Once', '本': 'this', '杀': 'kill', '极': 'pole',
            '查': 'check', '栗': 'chestnut', '株': 'stock', '样': 'kind', '检': 'Check',
            '欢': 'Happy', '死': 'dead', '汉': 'Chinese', '没': 'No', '治': 'rule',
            '法': 'law', '活': 'live', '点': 'point', '燻': 'Moth', '物': 'object',
            '猜': 'guess', '猴': 'monkey', '理': 'Rational', '生': 'Health', '用': 'use',
            '白': 'White', '百': 'hundred', '直': 'straight', '相': 'phase', '看': 'Look',
            '督': 'Supervisor', '知': 'know', '社': 'Society', '祝': 'wish', '积': 'product',
            '稣': 'Jesus', '经': 'through', '结': 'Knot', '给': 'give', '美': 'nice',
            '耶': 'Yay', '聊': 'chat', '胜': 'Win', '至': 'to', '虚': 'Virtual', '製': 'Made',
            '要': 'Want', '认': 'recognize', '讨': 'discuss', '让': 'Let', '识': 'knowledge',
            '话': 'words', '语': 'language', '说': 'Say', '谊': 'friendship',
            '谓': 'Predicate', '象': 'Elephant', '贺': 'He', '赢': 'win', '迎': 'welcome',
            '还': 'also', '这': 'This', '通': 'through', '鉄': 'iron', '问': 'ask',
            '阿': 'A', '题': 'question', '额': 'amount', '鬼': 'ghost', '鸡': 'Chicken',
            '가': 'end', '갈': 'Go', '게': 'to', '격': 'case', '경': 'circa', '관': 'tube',
            '국': 'soup', '금': 'gold', '나': 'I', '는': 'The', '니': 'Nee', '다': 'All',
            '대': 'versus', '도': 'Degree', '된': 'The', '드': 'De', '들': 'field',
            '때': 'time', '런': 'Run', '렵': 'Hi', '록': 'rock', '뤼': 'Crown',
            '리': 'Lee', '마': 'hemp', '만': 'just', '반': 'half', '분': 'minute',
            '사': 'four', '상': 'Prize', '서': 'book', '석': 'three', '성': 'castle',
            '스': 'The', '시': 'city', '않': 'Not', '야': 'Hey', '약': 'about',
            '어': 'uh', '와': 'Wow', '용': 'for', '유': 'U', '을': 'of', '이': 'this',
            '인': 'sign', '잘': 'well', '제': 'My', '쥐': 'rat', '지': 'G', '초': 'second',
            '캐': 'Can', '탱': 'Tang', '트': 'The', '티': 'tea', '패': 'tile', '품': 'Width',
            '한': 'One', '합': 'synthesis', '해': 'year', '허': 'Huh', '화': 'anger', '황': 'sulfur',
            '하': 'Ha', 'ﬁ': 'be', '０': '#', '２': '#', '８': '#', 'Ｅ': 'e', 'Ｇ': 'g',
            'Ｈ': 'h', 'Ｍ': 'm', 'Ｎ': 'n', 'Ｏ': 'O', 'Ｓ': 's', 'Ｕ': 'U', 'Ｗ': 'w',
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g',
            'ｈ': 'h', 'ｉ': 'i', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
            'ｒ': 'r', 'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｙ': 'y',
            '𝐀': 'a', '𝐂': 'c', '𝐃': 'd', '𝐅': 'f', '𝐇': 'h', '𝐊': 'k', '𝐍': 'n',
            '𝐎': 'o', '𝐑': 'r', '𝐓': 't', '𝐔': 'u', '𝐘': 'y', '𝐙': 'z', '𝐚': 'a',
            '𝐛': 'b', '𝐜': 'c', '𝐝': 'd', '𝐞': 'e', '𝐟': 'f', '𝐠': 'g', '𝐡': 'h',
            '𝐢': 'i', '𝐣': 'j', '𝐥': 'i', '𝐦': 'm', '𝐧': 'n', '𝐨': 'o', '𝐩': 'p',
            '𝐪': 'q', '𝐫': 'r', '𝐬': 's', '𝐭': 't', '𝐮': 'u', '𝐯': 'v', '𝐰': 'w',
            '𝐱': 'x', '𝐲': 'y', '𝐳': 'z', '𝑥': 'x', '𝑦': 'y', '𝑧': 'z', '𝑩': 'b',
            '𝑪': 'c', '𝑫': 'd', '𝑬': 'e', '𝑭': 'f', '𝑮': 'g', '𝑯': 'h', '𝑰': 'i',
            '𝑱': 'j', '𝑲': 'k', '𝑳': 'l', '𝑴': 'm', '𝑵': 'n', '𝑶': '0', '𝑷': 'p',
            '𝑹': 'r', '𝑺': 's', '𝑻': 't', '𝑾': 'w', '𝒀': 'y', '𝒁': 'z', '𝒂': 'a',
            '𝒃': 'b', '𝒄': 'c', '𝒅': 'd', '𝒆': 'e', '𝒇': 'f', '𝒈': 'g', '𝒉': 'h',
            '𝒊': 'i', '𝒋': 'j', '𝒌': 'k', '𝒍': 'l', '𝒎': 'm', '𝒏': 'n', '𝒐': 'o',
            '𝒑': 'p', '𝒒': 'q', '𝒓': 'r', '𝒔': 's', '𝒕': 't', '𝒖': 'u', '𝒗': 'v',
            '𝒘': 'w', '𝒙': 'x', '𝒚': 'y', '𝒛': 'z', '𝒩': 'n', '𝒶': 'a', '𝒸': 'c',
            '𝒽': 'h', '𝒾': 'i', '𝓀': 'k', '𝓁': 'l', '𝓃': 'n', '𝓅': 'p', '𝓇': 'r',
            '𝓈': 's', '𝓉': 't', '𝓊': 'u', '𝓌': 'w', '𝓎': 'y', '𝓒': 'c', '𝓬': 'c',
            '𝓮': 'e', '𝓲': 'i', '𝓴': 'k', '𝓵': 'l', '𝓻': 'r', '𝓼': 's', '𝓽': 't',
            '𝓿': 'v', '𝕴': 'j', '𝕸': 'm', '𝕿': 'i', '𝖂': 'm', '𝖆': 'a', '𝖇': 'b',
            '𝖈': 'c', '𝖉': 'd', '𝖊': 'e', '𝖋': 'f', '𝖌': 'g', '𝖍': 'h', '𝖎': 'i',
            '𝖒': 'm', '𝖓': 'n', '𝖕': 'p', '𝖗': 'r', '𝖘': 's', '𝖙': 't', '𝖚': 'u',
            '𝖛': 'v', '𝖜': 'w', '𝖞': 'n', '𝖟': 'z', '𝗕': 'b', '𝗘': 'e', '𝗙': 'f',
            '𝗞': 'k', '𝗟': 'l', '𝗠': 'm', '𝗢': 'o', '𝗤': 'q', '𝗦': 's', '𝗧': 't',
            '𝗪': 'w', '𝗭': 'z', '𝗮': 'a', '𝗯': 'b', '𝗰': 'c', '𝗱': 'd', '𝗲': 'e',
            '𝗳': 'f', '𝗴': 'g', '𝗵': 'h', '𝗶': 'i', '𝗷': 'j', '𝗸': 'k', '𝗹': 'i',
            '𝗺': 'm', '𝗻': 'n', '𝗼': 'o', '𝗽': 'p', '𝗿': 'r', '𝘀': 's', '𝘁': 't',
            '𝘂': 'u', '𝘃': 'v', '𝘄': 'w', '𝘅': 'x', '𝘆': 'y', '𝘇': 'z', '𝘐': 'l',
            '𝘓': 'l', '𝘖': 'o', '𝘢': 'a', '𝘣': 'b', '𝘤': 'c', '𝘥': 'd', '𝘦': 'e',
            '𝘧': 'f', '𝘨': 'g', '𝘩': 'h', '𝘪': 'i', '𝘫': 'j', '𝘬': 'k', '𝘮': 'm',
            '𝘯': 'n', '𝘰': 'o', '𝘱': 'p', '𝘲': 'q', '𝘳': 'r', '𝘴': 's', '𝘵': 't',
            '𝘶': 'u', '𝘷': 'v', '𝘸': 'w', '𝘹': 'x', '𝘺': 'y', '𝘼': 'a', '𝘽': 'b',
            '𝘾': 'c', '𝘿': 'd', '𝙀': 'e', '𝙃': 'h', '𝙅': 'j', '𝙆': 'k', '𝙇': 'l',
            '𝙈': 'm', '𝙊': 'o', '𝙋': 'p', '𝙍': 'r', '𝙏': 't', '𝙒': 'w', '𝙔': 'y',
            '𝙖': 'a', '𝙗': 'b', '𝙘': 'c', '𝙙': 'd', '𝙚': 'e', '𝙛': 'f', '𝙜': 'g',
            '𝙝': 'h', '𝙞': 'i', '𝙟': 'j', '𝙠': 'k', '𝙢': 'm', '𝙣': 'n', '𝙤': 'o',
            '𝙥': 'p', '𝙧': 'r', '𝙨': 's', '𝙩': 't', '𝙪': 'u', '𝙫': 'v', '𝙬': 'w',
            '𝙭': 'x', '𝙮': 'y', '𝟎': '0', '𝟏': '1', '𝟐': '2', '𝟓': '5', '𝟔': '6',
            '𝟖': '8', '𝟬': '0', '𝟭': '1', '𝟮': '2', '𝟯': '3', '𝟰': '4', '𝟱': '5',
            '𝟲': '6', '𝟳': '7', '𝟑': '3', '𝟒': '4', '𝟕': '7', '𝟗': '9',
            '🇦': 'a', '🇩': 'd', '🇪': 'e', '🇬': 'g', '🇮': 'i',
            '🇳': 'n', '🇴': 'o', '🇷': 'r', '🇹': 't', '🇼': 'w', '🖒': 'thumps up',
            'ℏ': 'h', 'ʲ': 'j', 'Ｃ': 'c', 'ĺ': 'i', 'Ｊ': 'j', 'ĸ': 'k', 'Ｐ': 'p'}

        self.puncts = ['_', '!', '?', '\x08', '\n', '\x0b', '\r', '\x10', '\x13', '\x1f', ' ', ' # ', '"', '#',
                       '# ', '$', '%', '&', '(', ')', '*', '+', ',', '/', '.', ':', ';', '<',
                       '=', '>', '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', '\x7f', '\x80',
                       '\x81', '\x85', '\x91', '\x92', '\x95', '\x96', '\x9c', '\x9d', '\x9f', '\xa0',
                       '¡', '¢༼', '£', '¤', '¥', '§', '¨', '©', '«', '¬', '\xad', '¯', '°', '±', '³',
                       '¶', '·', '¸', 'º', '»', '¼', '½', '¾', '¿', '×', 'Ø', '÷', 'ø', 'Ƅ', 'ƽ',
                       'ǔ', 'Ȼ', 'ɜ', 'ɩ', 'ʃ', 'ʌ', 'ʻ', 'ʼ', 'ˈ', 'ˌ', 'ː', '˙', '˚', '́', '̄', '̅',
                       '̇', '̈', '̣', '̨', '̯', '̱', '̲', '̶', '͜', '͝', '͞', '͟', '͡', 'ͦ', '؟', 'َ', 'ِ', 'ڡ',
                       '۞', '۩', '܁', 'ा', '्', 'ા', 'ી', 'ુ', '๏', '๏̯͡', '༼', '༽', 'ᐃ', 'ᐣ', 'ᐦ', 'ᐧ',
                       'ᑎ', 'ᑭ', 'ᑯ', 'ᒧ', 'ᓀ', 'ᓂ', 'ᓃ', 'ᓇ', 'ᔭ', 'ᴦ', 'ᴨ', 'ᵻ', 'Ἰ', 'Ἱ', 'ὼ',
                       '᾽', 'ῃ', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006',
                       '\u2007', '\u2008', '\u2009', '\u200a', '\u200b', '\u200c', '\u200d', '\u200e',
                       '\u200f', '‐', '‑', '‒', '–', '—', '―', '‖', '‘', '’', '‚', '‛', '“', '”', '„',
                       '†', '‡', '•', '‣', '…', '\u2028', '\u202a', '\u202c', '\u202d', '\u202f', '‰',
                       '′', '″', '‹', '›', '‿', '⁄', '⁍̴̛\u3000', '⁎', '⁴', '₂', '€', '₵', '₽', '℃', '℅',
                       'ℐ', '™', '℮', '⅓', '←', '↑', '→', '↓', '↳', '↴', '↺', '⇌', '⇒', '⇤', '∆', '∎',
                       '∏', '−', '∕', '∙', '√', '∞', '∩', '∴', '∵', '∼', '≈', '≠', '≤', '≥', '⊂', '⊕',
                       '⊘', '⋅', '⋆', '⌠', '⎌', '⏖', '─', '━', '┃', '┈', '┊', '┗', '┣', '┫', '┳', '╌', '═',
                       '║', '╔', '╗', '╚', '╣', '╦', '╩', '╪', '╭', '╭╮', '╮', '╯', '╰', '╱', '╲', '▀',
                       '▂', '▃', '▄', '▅', '▆', '▇', '█', '▊', '▋', '▏', '░', '▒', '▓', '▔', '▕',
                       '▙', '■', '▪', '▬', '▰', '▱', '▲', '▷', '▸', '►', '▼', '▾', '◄', '◇', '○',
                       '●', '◐', '◔', '◕', '◝', '◞', '◡', '◦', '★', '☆', '☏', '☐', '☒', '☙', '☛',
                       '☜', '☞', '☭', '☻', '☼', '♦', '♩', '♪', '♫', '♬', '♭', '♲', '⚆', '⚭', '⚲', '✀',
                       '✓', '✘', '✞', '✧', '✬', '✭', '✰', '✾', '❆', '❧', '➤', '➥', '⠀', '⤏', '⦁',
                       '⩛', '⬭', '⬯', '\u3000', '、', '。', '《', '》', '「', '」', '〔', '・', 'ㄸ', 'ㅓ',
                       '锟', 'ꜥ', '\ue014', '\ue600', '\ue602', '\ue607', '\ue608', '\ue613', '\ue807',
                       '\uf005', '\uf020', '\uf04a', '\uf04c', '\uf070', '\uf202\uf099', '\uf203',
                       '\uf071\uf03d\uf031\uf02f\uf032\uf028\uf070\uf02f\uf032\uf02d\uf061\uf029',
                       '\uf099', '\uf09a', '\uf0a7', '\uf0b7', '\uf0e0', '\uf10a', '\uf202',
                       '\uf203\uf09a', '\uf222', '\uf222\ue608', '\uf410', '\uf410\ue600', '\uf469',
                       '\uf469\ue607', '\uf818', '﴾', '﴾͡', '﴿', 'ﷻ', '\ufeff', '！', '％', '＇',
                       '（', '）', '，', '－', '．', '／', '：', '＞', '？', '＼', '｜', '￦', '￼', '�',
                       '𝒻', '𝕾', '𝖄', '𝖐', '𝖑', '𝖔', '𝗜', '𝘊', '𝘭', '𝙄', '𝙡', '𝝈', '🖑', '🖒']

    def replace_contractions(self, text, lower=False):
        """
        This functions check's whether a text contains contractions or not.
        In case a contraction is found, the corrected value from the dictionary is
        returned.
        Example: "I've" towards "I have"
        """

        # replace words with contraction according to the contraction_dict
        if lower:
            contraction_dict = self.contraction_dict_lower
        else:
            contraction_dict = self.contraction_dict

        if text.strip() in contraction_dict.keys():
            text = contraction_dict[text.strip()]

        # replace words with "'ve" to "have"
        matches = re.findall(r'\b\w+[\'`´]ve\b', text)
        if len(matches) != 0:
            text = re.sub(r'[\'`´]ve\b', " have", text)

        # replace words with "'re" to "are"
        matches = re.findall(r'\b\w+[\'`´]re\b', text)
        if len(matches) != 0:
            text = re.sub(r'[\'`´]re\b', " are", text)

        # replace words with "'ll" to "will"
        matches = re.findall(r'\b\w+[\'`´]ll\b', text)
        if len(matches) != 0:
            text = re.sub(r'[\'`´]ll\b', " will", text)

        # replace words with "'m" to "am"
        matches = re.findall(r'\b\w+[\'`´]m\b', text)
        if len(matches) != 0:
            text = re.sub(r'[\'`´]m\b', " am", text)

        # replace words with "'d" to "would"
        matches = re.findall(r'\b\w+[\'`´]d\b', text)
        if len(matches) != 0:
            text = re.sub(r'[\'`´]d\b', " would", text)

        # replace all "'s" by space
        matches = re.findall(r'\b\w+[\'`´]s\b', text)
        if len(matches) != 0:
            text = re.sub(r'[\'`´]s\b', " ", text)

        return text

    def replace_symbol_special(self, text, check_vocab=False, vocab=None):
        '''
        This method can be used to replace dashes ('-') around and within the words using regex.
        It only removes dashes for words which are not known to the vocabluary.
        Next to that, common word separators like underscores ('_') and slashes ('/') are replaced by spaces.
        '''

        # replace all dashes and abostropes at the beginning of a word with a space
        matches = re.findall(r"\s+(?:-|')\w*", text)
        # if there is a match is in text
        if len(matches) != 0:
            # remove the dash from the match or better text
            for match in matches:
                text = re.sub(match, re.sub(r"(?:-|')", ' ', match), text)

        # replace all dashes and abostrophes at the end of a word with a space
        # function works as above
        matches = re.findall(r"\w*(?:-|')\s+", text)
        if len(matches) != 0:
            for match in matches:
                text = re.sub(match, re.sub(r"(?:-|')", ' ', match), text)

        if check_vocab == True:
            # replace dashes and abostrophes in the middle of the word only in case it is not known to a dictionary
            # function works as above
            matches = re.findall(r"\w*(?:-|')\w*", text)
            if len(matches) != 0:
                for match in matches:
                    # check if the word with dash in the middle in in the vocabluary
                    if match not in vocab:
                        text = re.sub(match, re.sub(r"(?:-|')", ' ', match), text)

        #
        text = re.sub(r'(?:_|\/)', ' ', text)

        text = re.sub(r' +', ' ', text)  # -
        return text

    def find_smilies(self, text):
        '''
        For investigation only: Find most common keyboard typed smilies in the text.
        '''

        # define a pattern to find typical keyboard smilies
        pattern = r"((?:3|<)?(?::|;|=|B)(?:-|'|'-)?(?:\)|D|P|\*|\(|o|O|\]|\[|\||\\|\/)\s)"
        # Find the matches n the text
        matches = re.findall(pattern, text)
        # If the text contain matches print the text and the smilies found
        if len(matches) != 0:
            print(text, matches)

    def replace_smilies(self, text):
        '''
        Simplyfied method to replace keyboard smilies with its very simple translation.
        '''

        # Find and replace all happy smilies
        matches = re.findall(r"((?:<|O|o|@)?(?::|;|=|B)(?:-|'|'-)?(?:\)|\]))", text)
        if len(matches) != 0:
            text = re.sub(r"((?:<|O|o|@)?(?::|;|=|B)(?:-|'|'-)?(?:\)|\]))", " smile ", text)

        # Find and replace all laughing smilies
        matches = re.findall(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:d|D|P|p)\b)", text)
        if len(matches) != 0:
            text = re.sub(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:d|D|P|p)\b)", " smile ", text)

        # Find and replace all unhappy smilies
        matches = re.findall(r"((?:3|<)?(?::|;|=|8)(?:-|'|'-)?(?:\(|\[|\||\\|\/))", text)
        if len(matches) != 0:
            text = re.sub(r"((?:3|<)?(?::|;|=|8)(?:-|'|'-)?(?:\(|\[|\||\\|\/))", " unhappy ", text)

        # Find and replace all kissing smilies
        matches = re.findall(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:\*))", text)
        if len(matches) != 0:
            text = re.sub(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:\*))", " kiss ", text)

        # Find and replace all surprised smilies
        matches = re.findall(r"((?::|;|=)(?:-|'|'-)?(?:o|O)\b)", text)
        if len(matches) != 0:
            text = re.sub(r"((?::|;|=)(?:-|'|'-)?(?:o|O)\b)", " surprised ", text)

        # Find and replace all screaming smilies
        matches = re.findall(r"((?::|;|=)(?:-|'|'-)?(?:@)\b)", text)
        if len(matches) != 0:
            text = re.sub(r"((?::|;|=)(?:-|'|'-)?(?:@)\b)", " screaming ", text)

        # Find and replace all hearts
        matches = re.findall(r"♥|❤|<3|❥|♡", text)
        if len(matches) != 0:
            text = re.sub(r"(?:♥|❤|<3|❥|♡)", " love ", text)

        text = re.sub(' +', ' ', text)
        return text

    def remove_stopwords(self, text, stop_words):
        '''
        Remove stopwords and multiple whitespaces around words
        '''

        # Compile stopwords separated by | and stopped by word boundary
        stopword_re = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b')
        # Replace the stopwords by space
        text = stopword_re.sub(' ', text)
        # Replace double spaces by a single space
        text = re.sub(' +', ' ', text)
        return text

    def clean_text(self, text, scope='general'):
        '''
        This function handles text cleaning from various symbols.
        - it translates special font types into the standard text type of python.
        - it removes all symbols except for dashes and abostrophes being handled by
          "replace_symbol_special".
        - it handles multi letter appearances like "comiiii" > "comi"
        - typical unknown words like "Trump"
        '''

        # compile all special symbols from the dictionary to one regex function
        translate_regex = re.compile(r'(' + r'|'.join(self.translate_dictionary.keys()) + r')')

        # find all matches of special symbols in the text
        matches = re.findall(translate_regex, text)
        # if there is one or more matches
        if len(matches) != 0:
            for x in matches:
                if x in self.translate_dictionary.keys():
                    # replace the symbol by its replacement item
                    text = re.sub(x, self.translate_dictionary.get(x), text)

        # find and remove all "http" links
        matches = re.findall(r'http\S+', text)
        if len(matches) != 0:
            text = re.sub(r'http\S+', '', text)

        # remove all backslashes
        matches = re.findall(r'\\', text)
        if len(matches) != 0:
            text = re.sub(r'\\', ' ', text)

        # compile all remaining special characters into one translate line and replace them by space
        # the translate function is really fast thus here our preferred choice
        text = text.translate(str.maketrans(''.join(self.puncts), len(''.join(self.puncts)) * ' '))

        # find words where 4 repetitions of a letter goes in a row and reduce them to only one
        # we are not correcting words with 2 or three identical letters in a row as this could destroy correct words
        # first find repeating characters
        matches = re.findall(r'(.)\1{3,}', text)
        # is some are found
        if len(matches) != 0:
            # for each match replace it with its first letter (x[0])
            for x in matches:
                character_re = re.compile(x + '{3,}')
                matchesInside = re.findall(character_re, text)
                if len(matchesInside) != 0:
                    for x in matchesInside:
                        text = re.sub(x, x[0], text)

        # hahaha s by one haha
        matches = re.findall(r'\b[h,a]{4,}\b', text)
        if len(matches) != 0:
            text = re.sub(r'\b[h,a]{4,}\b', 'haha', text)

        # as we found many unknown word variations including 'Trump' we reduce thse  words just to Trump
        # being represented in most word vectors
        matches = re.findall(r'\w*[Tt][Rr][uU][mM][pP]\w*', text)
        if len(matches) != 0:
            for x in matches:
                text = re.sub(x, 'Trump', text)

        # remove potential double spaces generated during processing
        text = re.sub(' +', ' ', text)

        # those symbols are not touched by this function ->see replace_contraction or replace_special_symbols
        # keep = ["'", '-', '´']


        return text

    def clean_numbers(self, x):
        """
        The following function is used to format the numbers.
        In the beginning "th, st, nd, rd" are removed
        """

        # remove "th" after a number
        matches = re.findall(r'\b\d+\s*th\b', x)
        if len(matches) != 0:
            x = re.sub(r'\s*th\b', " ", x)

        # remove "rd" after a number
        matches = re.findall(r'\b\d+\s*rd\b', x)
        if len(matches) != 0:
            x = re.sub(r'\s*rd\b', " ", x)

        # remove "st" after a number
        matches = re.findall(r'\b\d+\s*st\b', x)
        if len(matches) != 0:
            x = re.sub(r'\s*st\b', " ", x)

        # remove "nd" after a number
        matches = re.findall(r'\b\d+\s*nd\b', x)
        if len(matches) != 0:
            x = re.sub(r'\s*nd\b', " ", x)

        # replace standalone numbers higher than 10 by #
        # this function does not touch numbers linked to words like "G-20"
        matches = re.findall(r'^\d+\s+|\s+\d+\s+|\s+\d+$', x)
        if len(matches) != 0:
            x = re.sub('^[0-9]{5,}\s+|\s+[0-9]{5,}\s+|\s+[0-9]{5,}$', ' ##### ', x)
            x = re.sub('^[0-9]{4}\s+|\s+[0-9]{4}\s+|\s+[0-9]{4}$', ' #### ', x)
            x = re.sub('^[0-9]{3}\s+|\s+[0-9]{3}\s+|\s+[0-9]{3}$', ' ### ', x)
            x = re.sub('^[0-9]{2}\s+|\s+[0-9]{2}\s+|\s+[0-9]{2}$', ' ## ', x)
            # we do include the range from 1 to 10 as all word-vectors include them
            # x = re.sub('[0-9]{1}', '#', x)

        return x

    def year_and_hour(self, text):
        """
        This function is used to replace "yr,yrs" by year and "hr,hrs" by hour.
        """

        # Find matches for "yr", "yrs", "hr", "hrs"
        matches_year = re.findall(r'\b\d+\s*yr\b', text)
        matches_years = re.findall(r'\b\d+\s*yrs\b', text)
        matches_hour = re.findall(r'\b\d+\s*hr\b', text)
        matches_hours = re.findall(r'\b\d+\s*hrs\b', text)

        # replace all matches accordingly
        if len(matches_year) != 0:
            text = re.sub(r'\b\d+\s*yr\b', "year", text)
        if len(matches_years) != 0:
            text = re.sub(r'\b\d+\s*yrs\b', "year", text)
        if len(matches_hour) != 0:
            text = re.sub(r'\b\d+\s*hr\b', "hour", text)
        if len(matches_hours) != 0:
            text = re.sub(r'\b\d+\s*hrs\b', "hour", text)
        return text

    def build_vocab(self, df):
        '''Build a dictionary of words and its number of occurences from the data frame'''

        # initialize the tokenizer
        tokenizer = TweetTokenizer()

        vocab = {}
        for i, row in enumerate(df):
            words = tokenizer.tokenize(row)
            for w in words:
                try:
                    vocab[w] += 1
                except KeyError:
                    vocab[w] = 1

        return vocab

    def check_coverage(self, vocab, embeddings_index):
        known_words = {}
        unknown_words = {}
        nb_known_words = 0
        nb_unknown_words = 0
        for word in vocab.keys():
            if (word in embeddings_index) or (word.lower() in embeddings_index) or (word.title() in embeddings_index):
                known_words[word] = vocab[word]
                nb_known_words += vocab[word]
            else:
                unknown_words[word] = vocab[word]
                nb_unknown_words += vocab[word]

        print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
        print('Found embeddings for {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
        # unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

        return unknown_words

    def process(self, text, model_vocab, lower=False):
        if lower:
            text = text.str.lower()
        tokenizer = TweetTokenizer()
        vocab = self.build_vocab(text)
        unknown = self.check_coverage(vocab, model_vocab, )
        unknown = unknown.keys()

        corrected = [self.replace_contractions(x, lower) for x in unknown]
        corrected = [emoji.demojize(x) for x in corrected]
        corrected = [self.replace_smilies(x) for x in corrected]
        corrected = [self.clean_text(x) for x in corrected]
        corrected = [self.clean_numbers(x) for x in corrected]
        corrected = [self.replace_symbol_special(x, check_vocab=True, vocab=model_vocab) for x in corrected]
        corrected = [self.year_and_hour(x) for x in corrected]
        dictionary = dict(zip(unknown, corrected))

        # remove all keys where unknown equals correction after processing
        # create a new dict
        dict_mispell = dict()
        for key in dictionary.keys():
            # if the correction differs from the unknown word add it to the new dict
            if key != dictionary.get(key):
                dict_mispell[key] = dictionary.get(key)

        def clean_mispell(text, dict_mispell):
            # tokenize the text with TweetTokenizer
            words = tokenizer.tokenize(text)
            for i, word in enumerate(words):
                if word in dict_mispell.keys():
                    words[i] = dict_mispell.get(word)

            text = ' '.join(words)
            text = re.sub(r' +', ' ', text)
            return text

        text = text.apply(lambda x: clean_mispell(x, dict_mispell))

        return text

    def build_matrix(self, word_index, embedding):
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        unknown_words = []

        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding[word.title()]
                    except KeyError:
                        unknown_words.append(word)
        return embedding_matrix, unknown_words


##############################################################
#                          Utils                             #
##############################################################

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def generate_kfold(cv, train_num):
    ids = list(range(train_num))
    kfolder = KFold(n_splits=cv, shuffle=True, random_state=1993)

    kfold = []
    for train_idx, validate_idx in kfolder.split(ids):
        kfold.append([train_idx, validate_idx])

    with open('kfold_{}.pkl'.format(cv), 'wb') as f:
        pickle.dump(kfold, f)


def config2str(config):
    string = '\n'
    for k, v in sorted(config.items(), key=lambda x: x[0]):
        string += '{}:\t{}\n'.format(k, v)
    return string


def load_config(log_path):
    config = {}
    with open(log_path, 'r') as f:
        for line in f.readlines():
            if ':' in line and not ('Fold' in line):
                k = line.split(':')[0].strip()
                if k in ['output_feature_num', 'num_targets', 'gpu', 'epoch_num', 'epoch_num', 'embedding_dim']:
                    v = int(line.split(':')[1].strip())
                elif k in ['model_name', 'text']:
                    v = line.split(':')[1].strip()
                else:
                    v = float(line.split(':')[1].strip())
                config[k] = v
    return config


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def bert_convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (
        max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def get_statistics_feature(text, toxicity_word, unknow):
    if isinstance(text, pd.Series):
        text = pd.DataFrame(text)
    text['comment_text'] = text['comment_text'].str.lower()
    text['split'] = text['comment_text'].apply(str.split)
    text['total_length'] = text['comment_text'].apply(len)
    text['capitals'] = text['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    text['caps_vs_length'] = text['capitals'] / text['total_length']
    text['num_exclamation_marks'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '!！'))
    text['num_question_marks'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '?？'))
    text['num_punctuation'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    text['num_symbols'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '&$%'))
    text['num_star'] = text['comment_text'].apply(lambda comment: comment.count('*'))
    text['num_words'] = text['split'].apply(len)
    text['num_unique_words'] = text['split'].apply(lambda comment: len(set(comment)))
    text['num_smilies'] = text['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    text['num_toxicity'] = text['split'].apply(lambda comment: len(list(filter(lambda w: w in toxicity_word, comment))))
    text['num_unknow'] = text['split'].apply(lambda comment: len(list(filter(lambda w: w in unknow, comment))))
    text['unique_vs_words'] = text['num_unique_words'] / text['num_words']
    text['toxicity_vs_words'] = text['num_toxicity'] / text['num_words']
    text['unknow_vs_words'] = text['num_unknow'] / text['num_words']

    feature_columns = ['total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks', 'num_question_marks',
                       'num_punctuation',
                       'num_symbols', 'num_star', 'num_words', 'num_unique_words', 'num_smilies', 'num_toxicity',
                       'num_unknow',
                       'unique_vs_words', 'toxicity_vs_words', 'unknow_vs_words']

    return text[feature_columns]

##############################################################
#                      Dataset Helper                        #
##############################################################
class ToxicityDataset(Dataset):
    def __init__(self,x,y,weight,idx):
        super(ToxicityDataset,self).__init__()

        x = np.asarray(x)
        self.x = x[idx]
        self.y = y[idx]
        self.idx = idx
        self.weight = weight[idx]

    def __getitem__(self, id):
        return self.x[id], self.y[id], self.weight[id], self.idx[id]

    def __len__(self):
        return len(self.idx)

class ToxicityTestDataset(Dataset):
    def __init__(self,x,ids):
        super(ToxicityTestDataset,self).__init__()

        self.x = x
        self.ids = ids

        assert len(x) == len(ids)

    def __getitem__(self, id):
        return self.x[id],self.ids[id]

    def __len__(self):
        return len(self.ids)

class SequenceBucketCollator():
    def __init__(self,percentile=100):
        self.percentile = percentile

    def __call__(self, batch):
        batch_ = list(zip(*batch))
        data_num = len(batch[0])
        assert data_num >= 1

        lens = [len(x) for x in batch_[0]]
        max_len_ = min(np.percentile(lens, self.percentile),256)

        batch_[0] = sequence.pad_sequences(batch_[0], maxlen=int(max_len_))
        batch_[0] = np.asarray(batch_[0],dtype=np.int)

        for i in range(1,data_num):
            batch_[i] = np.asarray(batch_[i])

        return batch_

##############################################################
#                      JigsawEvaluator                       #
##############################################################

class JigsawEvaluator:

    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score

##############################################################
#                         Ensembler                          #
##############################################################

class Ensembler():
    def __init__(self,model_num,):
        self.model_num = model_num
        self.max_scroe = 0
        self.stop_round = 10
        self.sample_num = min(2,model_num)
        self.weight = np.ones(model_num,)
        # self.resolution = 0.5

    @staticmethod
    def softmax(x):
        return np.exp(x)/np.exp(x).sum()

    def sample(self,list,num):
        return [random.choice(list) for _ in range(num)]

    def evaluate(self,pred, y_train, y_identity,):
        jigsawevaluator = utils.JigsawEvaluator(y_train, y_identity)
        return jigsawevaluator.get_final_metric(pred)

    # random search, until score stop imporve for n round
    def fit(self, X, y_train, y_identity,):
        assert X.shape[1] == self.model_num
        jigsawevaluator = utils.JigsawEvaluator(y_train, y_identity)
        round = 0
        while round < self.stop_round:
            pre_weight = self.weight.copy()
            idx = self.sample(list(range(self.model_num)),self.sample_num)
            op = self.sample([1,-1],self.sample_num)
            for i in range(self.sample_num):
                self.weight[idx[i]] += op[i] * random.random()
            weights = self.softmax(self.weight) # make sure sum==1

            res = (weights * X).sum(axis=1)
            score = jigsawevaluator.get_final_metric(res)
            if score > self.max_scroe:
                self.max_scroe = score
                round = 0
                print('Inprove! Score:{:.6f}\tWeight:{}'.format(self.max_scroe,self.softmax(self.weight)))
            else:
                self.weight = pre_weight.copy()
                round += 1
        print('Ensemble Done! \nFinal Score:{}\nFinal Weight:{}\n'.format(self.max_scroe,self.softmax(self.weight)))
        return self.max_scroe

    def predict(self,X):
        assert X.shape[1] == self.model_num
        return (self.softmax(self.weight) * X).sum(axis=1)

    def save(self,path):
        dump_ = self.__dict__
        with open(path,'wb') as f:
            pickle.dump(dump_,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            parameters = pickle.load(f)
        model = Ensembler(parameters['model_num'])
        model.weight = parameters['weight']
        model.stop_round = parameters['stop_round']
        model.max_scroe = parameters['max_scroe']
        model.sample_num = parameters['sample_num']
#         model.resolution = parameters['resolution']

        return model

##############################################################
#                         DL Models                          #
##############################################################
def load_model(model_name,num_targets, embedding_matrix=None, embedding_size=None, output_feature_num=50,**kw):
    if model_name == 'Toxicity_LSTM2':
        return Toxicity_LSTM2(num_targets,embedding_matrix,embedding_size,output_feature_num)
    elif model_name == 'Toxicity_BiLSTMSelfAttention':
        return Toxicity_BiLSTMSelfAttention(num_targets,embedding_matrix,embedding_size,output_feature_num)
    else:
        raise Exception("Error model name:{}".format(model_name))

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class Toxicity_LSTM2(nn.Module):
    def __init__(self, num_targets, embedding_matrix=None, embedding_size=None, output_feature_num=50, lstm_middle=128):
        super(Toxicity_LSTM2, self).__init__()

        self.embedding = nn.Embedding(*embedding_size)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.output_feature_num = output_feature_num
        self.lstm_middle = lstm_middle
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embedding_size[1], lstm_middle, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_middle * 2, lstm_middle, bidirectional=True, batch_first=True)

        self.fc1_1 = nn.Linear(lstm_middle*6, lstm_middle*6)
        self.fc1_2 = nn.Linear(lstm_middle*6, lstm_middle*6)

        self.fc2 = nn.Linear(lstm_middle*6, output_feature_num)
        self.fc3 = nn.Linear(output_feature_num, num_targets)

    def cuda_(self,gpu):
        self.cuda(gpu)
        self.embedding = self.embedding.cuda(gpu)

    def forward(self, x):

        x = self.embedding(x)
        x = self.embedding_dropout(x).float()

        h_lstm1, _ = self.lstm1(x)
        out, (hn,_) = self.lstm2(h_lstm1)

        hn = hn.transpose(0,1).contiguous().view(-1,2*self.lstm_middle)
        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)

        out = torch.cat((hn, max_pool, avg_pool), 1)
        out1 = F.relu(self.fc1_1(out))
        out2 = F.relu(self.fc1_2(out))
        out = out + out1 + out2

        feature = self.fc2(out)
        out = torch.sigmoid(self.fc3(F.relu(feature)))

        return out,feature

    @property
    def name(self):
        return 'Toxicity_LSTM2'


class Toxicity_BiLSTMSelfAttention(nn.Module):
    def __init__(self, num_targets, embedding_matrix=None, embedding_size=None, output_feature_num=200,lstm_h=64,attn_d=64):
        super(Toxicity_BiLSTMSelfAttention, self).__init__()

        self.embedding = nn.Embedding(*embedding_size)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.output_feature_num = output_feature_num
        self.lstm = nn.LSTM(embedding_size[1],lstm_h,bidirectional=True,batch_first=True)

        self.d = lstm_h * 2
        self.n_head = 4
        self.d_q = attn_d
        self.d_k = attn_d
        self.d_v = attn_d
        self.temperature = np.power(self.d_k, 0.5)
        assert self.d_q == self.d_k

        self.fc_q = nn.Linear(self.d,self.d_q * self.n_head)
        self.fc_k = nn.Linear(self.d,self.d_k * self.n_head)
        self.fc_v = nn.Linear(self.d,self.d_v * self.n_head)

        self.fc = nn.Linear(self.n_head*self.d_v,128)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, output_feature_num)
        self.fc3 = nn.Linear(output_feature_num,num_targets)

        self.layer_norm = nn.LayerNorm(self.d)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc_q.weight, mean=0, std=np.sqrt(2.0 / (self.d + self.d_q)))
        nn.init.normal_(self.fc_k.weight, mean=0, std=np.sqrt(2.0 / (self.d + self.d_k)))
        nn.init.normal_(self.fc_v.weight, mean=0, std=np.sqrt(2.0 / (self.d + self.d_v)))

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def cuda_(self,gpu):
        self.cuda(gpu)
        self.embedding = self.embedding.cuda(gpu)

    def forward(self, x,):
        bs = x.shape[0]
        l = x.shape[1]

        mask = (x == 0)  # (N, L,)
        # make sure attn wont be nan
        mask[:,-1] = 0
        mask = mask.unsqueeze(1).repeat(self.n_head,l,1)

        # embbeding
        x = self.embedding(x)
        x = self.embedding_dropout(x).float()

        # lstm
        x,_ = self.lstm(x)

        # self-attention
        q = self.fc_q(x).view(bs,l,self.n_head,self.d_q)
        k = self.fc_k(x).view(bs,l,self.n_head,self.d_k)
        v = self.fc_v(x).view(bs,l,self.n_head,self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l, self.d_q) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l, self.d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l, self.d_v) # (n*b) x lv x dv

        attn = torch.bmm(q,k.transpose(1,2))
        attn /= self.temperature
        attn = attn.masked_fill(mask, -np.inf)
        attn = F.softmax(attn,dim=2)

        x = torch.bmm(attn,v)

        x = x.view(self.n_head,bs,l,self.d_v)
        x = x.permute(1, 2, 0, 3).contiguous().view(bs, l, -1)  # b x lq x (n*dv)

        x = self.fc(x)

        # represent
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        x = torch.cat((max_pool, avg_pool), 1)

        x = F.relu(self.fc1(x))
        feature = self.fc2(x)

        out = torch.sigmoid(self.fc3(F.relu(feature)))

        return out,feature

class Toxicity_NN(nn.Module):
    def __init__(self,feature_num,target_num,hidden=1024):
        super(Toxicity_NN,self).__init__()
        self.fc1 = nn.Linear(feature_num,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,target_num)
        self.drop = nn.Dropout()

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

# global variable
cv = 3
max_len = 220
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
target_num = len(aux_columns) + 2
# test_csv_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
test_csv_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
kfold_path = '../input/toxicity-inference-pkl/kfold_3.pkl'
# glove_embedding_path = '../input/covert-glove-to-pkl-file/glove_embedding.pkl'
# crawl_embedding_path = '../input/covert-crawl-to-pkl-file/crawl_embedding.pkl'
glove_embedding_path = '../input/glove-embedding/glove_embedding.pkl'
crawl_embedding_path = '../input/crawl-embedding/crawl_embedding.pkl'

toxicity_word_path = '../input/toxicity-inference-pkl/toxicity_word.pkl'
statistics_normalize_path = '../input/toxicity-inference-pkl/normalize_data.pkl'

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
bert_vacab_path = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'
bert_config_path = '../input/toxicity-models/bert_2epoch_v1/bert_config_v1.json'


TEST_PRED = {}

##############################################################
#                                                            #
#                        Load Data                           #
#                                                            #
##############################################################

start_time = datetime.now()

nrows = None
test = pd.read_csv(test_csv_path, nrows=nrows)
test_id = test['id']
with open(kfold_path, 'rb') as f:
    kfold = pickle.load(f)
with open(glove_embedding_path, 'rb') as f:
    glove_embedding = pickle.load(f)
with open(crawl_embedding_path, 'rb') as f:
    crawl_embedding = pickle.load(f)
# glove_embedding = load_embeddings(glove_embedding_path)
# crawl_embedding = load_embeddings(crawl_embedding_path)

x_test = {'BERT': None, 'LSTM': None}

tp = Text_Process()
x_test_lstm = tp.process(test['comment_text'], set(glove_embedding.keys()) | set(crawl_embedding.keys()))
tokenizer = text.Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(list(x_test_lstm))
x_test_lstm = tokenizer.texts_to_sequences(x_test_lstm)
glove_matrix, unknown_words_glove = tp.build_matrix(tokenizer.word_index, glove_embedding)
crawl_matrix, unknown_words_crawl = tp.build_matrix(tokenizer.word_index, crawl_embedding)
embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
unknown_words = set(unknown_words_crawl + unknown_words_glove)
x_test['LSTM'] = x_test_lstm

del x_test_lstm
del glove_embedding
del crawl_embedding
del glove_matrix
del crawl_matrix
del unknown_words_glove
del unknown_words_crawl
gc.collect()

bert_vocab = load_vocab(bert_vacab_path)
x_test_bert = tp.process(test['comment_text'], bert_vocab.keys(), lower=True)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
x_test_bert = bert_convert_lines(x_test_bert.fillna("DUMMY_VALUE"), max_len, tokenizer)
x_test['BERT'] = x_test_bert

x_test_original = test['comment_text']

del test
del x_test_bert
del tokenizer
del bert_vocab
gc.collect()

print('Data Loaded!')
print('Use Time:', datetime.now() - start_time)

##############################################################
#                                                            #
#              Statistics Feature Extract                    #
#                                                            #
##############################################################
start_time = datetime.now()

with open(toxicity_word_path,'rb') as f:
    toxicity_word = pickle.load(f)['total']
with open(statistics_normalize_path,'rb') as f:
    statistics_normalize = pickle.load(f)

statistics_features = get_statistics_feature(x_test_original,toxicity_word,unknown_words)
# normalize
for c in statistics_features.columns:
    mean = statistics_normalize[c+'_mean']
    std = statistics_normalize[c+'_std']
    statistics_features[c] = statistics_features[c].fillna(mean)
    statistics_features[c] = (statistics_features[c] - mean) /std
statistics_features = statistics_features.values
statistics_features = np.repeat(statistics_features[np.newaxis, :, :], cv, axis=0)

del x_test_original
del toxicity_word
del unknown_words
gc.collect()

print('Statistics Feature Extracted! Shape:',statistics_features.shape)
print('Use Time:',datetime.now()-start_time)

##############################################################
#                                                            #
#                 DL Feature Extract                         #
#                                                            #
##############################################################
start_time = datetime.now()

extract_models = ['BERT_v1', 'BERT_v2', 'Toxicity_BiLSTMSelfAttention', 'Toxicity_LSTM2', ]
dl_features = []

# Load models
# Generate Features & Predictions
for model_name in extract_models:
    if 'BERT' in model_name:
        test_features = []
        test_preds = []
        for fold in range(cv):
            bert_config = BertConfig(bert_config_path)
            model = BertForSequenceClassification(bert_config, num_labels=target_num, feature_num=50)
            if model_name == 'BERT_v1':
                model_path = "../input/toxicity-models/bert_2epoch_v1/fold{}.bin".format(fold)
            else:
                model_path = "../input/toxicity-models/bert_2epoch_v2/fold{}.bin".format(fold)
            model.load_state_dict(torch.load(model_path))
            model.cuda()
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            test_dataset = TensorDataset(torch.tensor(x_test['BERT'], dtype=torch.long), )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

            test_pred_fold = []
            test_feature_fold = []

            for x in tqdm_notebook(test_loader):
                max_len_batch = torch.max(torch.sum((x[0] != 0), 1))
                x = x[0][:, :max_len_batch].long().cuda()
                pred, feature = model(x, attention_mask=(x > 0).cuda(), )
                test_feature_fold.append(feature)
                test_pred_fold.append(torch.sigmoid(pred[:, 0]))

            test_pred_fold = torch.cat(test_pred_fold, dim=0)
            test_preds.append(test_pred_fold.cpu().numpy())
            test_feature_fold = torch.cat(test_feature_fold, dim=0)
            test_features.append(test_feature_fold.cpu().numpy())

        test_features = np.asarray(test_features)
        dl_features.append(test_features)
        test_pred = np.asarray(test_preds).mean(axis=0)
        TEST_PRED[model_name] = test_pred

    else:
        test_features = []
        test_preds = []
        for fold in range(cv):
            model = load_model(model_name, target_num, embedding_matrix, embedding_matrix.shape, 50)
            model_dict = torch.load('../input/toxicity-models/{}_v1/fold{}.pth'.format(model_name.lower(), fold))
            model_dict['embedding.weight'] = torch.tensor(embedding_matrix)
            model.load_state_dict(model_dict)
            model.cuda()
            model.eval()

            test_dataset = ToxicityTestDataset(x_test['LSTM'], test_id)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                                         collate_fn=SequenceBucketCollator())

            test_pred_fold = []
            test_feature_fold = []
            test_id_fold = []

            with torch.no_grad():
                for x, ids in tqdm_notebook(test_dataloader):
                    x = torch.from_numpy(x).long().cuda()

                    pred, feature = model(x)
                    pred = pred[:, 0]

                    test_id_fold.extend(ids)
                    test_pred_fold.append(pred)
                    test_feature_fold.append(feature)

            test_pred_fold = torch.cat(test_pred_fold, dim=0)
            test_preds.append(test_pred_fold.cpu().numpy())
            test_feature_fold = torch.cat(test_feature_fold, dim=0)
            test_features.append(test_feature_fold.cpu().numpy())

        test_features = np.asarray(test_features)
        dl_features.append(test_features)
        test_pred = np.asarray(test_preds).mean(axis=0)
        TEST_PRED[model_name] = test_pred

dl_features = np.concatenate(dl_features, axis=2)

del embedding_matrix
del model
del test_dataset
del test_dataloader
del test_features
del test_preds
del x_test['BERT']
del x_test['LSTM']
del x_test
gc.collect()
print('Feature Extracted! Shape:', dl_features.shape)
print('Use Time:', datetime.now() - start_time)

##############################################################
#                                                            #
#                          Predict                           #
#                                                            #
##############################################################
start_time = datetime.now()

# NeuralNetwork, Lightgbm, RandomFroest
nn_path = '../input/toxicity-models/nn_v2'
lgb_path = '../input/toxicity-models/lgb_v2'
rf_path = '../input/toxicity-models/rf_v2'
xgb_path = '../input/toxicity-models/xgb_v2'

features = [statistics_features, dl_features]
features = np.concatenate(features, axis=2)
feature_num = features.shape[2]
# print(features.shape)

models = ['NN', 'LGB', 'RF', 'XGB']

for model_name in models:
    if model_name == 'NN':
        pred_test = []
        for fold in range(cv):
            model = Toxicity_NN(feature_num, target_num)
            model_dict = torch.load(os.path.join(nn_path, 'fold{}.pth'.format(fold)))
            model.load_state_dict(model_dict)
            model.cuda()
            model.eval()

            x_test = features[fold]
            test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float), )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
            pred_test_fold = []
            with torch.no_grad():
                for x, in tqdm(test_loader):
                    x = x.float().cuda()
                    pred = model(x)
                    pred_test_fold.append(pred[:, 0])
            pred_test_fold = torch.cat(pred_test_fold, dim=0)
            pred_test_fold = pred_test_fold.cpu().numpy()
            pred_test.append(pred_test_fold)

    elif model_name == 'LGB':
        pred_test = []
        for fold in range(cv):
            x_test = features[fold]
            clf = lgb.Booster(model_file='{}/fold{}.pth'.format(lgb_path, fold))
            pred_test_fold = clf.predict(x_test, num_iteration=clf.best_iteration)
            pred_test.append(pred_test_fold)

    elif model_name == 'RF':
        pred_test = []
        for fold in range(cv):
            x_test = features[fold]
            with open('{}/fold{}.pth'.format(rf_path, fold), 'rb') as f:
                clf = pickle.load(f)
            pred_test_fold = clf.predict_proba(x_test, )
            pred_test_fold = pred_test_fold[:, 1]
            pred_test.append(pred_test_fold)

    elif model_name == 'XGB':
        pred_test = []
        for fold in range(cv):
            x_test = features[fold]
            x_test = xgb.DMatrix(x_test)
            clf = xgb.Booster(model_file='{}/fold{}.pth'.format(xgb_path, fold))
            pred_test_fold = clf.predict(x_test)
            pred_test.append(pred_test_fold)

    else:
        pass

    pred_test = np.asarray(pred_test)
    pred_test = pred_test.mean(axis=0)
    TEST_PRED[model_name] = pred_test

print('Use Time:', datetime.now() - start_time)

##############################################################
#                                                            #
#                        Ensemble                            #
#                                                            #
##############################################################
ensembl_model_path = '../input/toxicity-models/ensemble_v2/ensembler.pth'
order = ['LGB','NN','RF','XGB','BERT_v1','BERT_v2','Toxicity_BiLSTMSelfAttention','Toxicity_LSTM2']

test_preds = []
for model_name in order:
    test_preds.append(TEST_PRED[model_name])
test_preds = np.asarray(test_preds).transpose((1,0))

model = Ensembler.load(ensembl_model_path)
test_pred = model.predict(test_preds)

test_pred = pd.DataFrame({'id': test_id, 'prediction': test_pred})
test_pred.set_index(keys='id',drop=True,inplace=True)
test_pred.to_csv('submission.csv')


