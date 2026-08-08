import re
import unicodedata

ALIASES = {
    "bolivia plurinational state of": "bolivia",
    "bosnia herzegovina": "bosnia and herzegovina",
    "brunei darussalam": "brunei",
    "china people's republic of": "china",
    "cote d ivoire": "ivory coast",
    "czech republic": "czechia",
    "democratic republic of the congo": "democratic republic of congo",
    "iran islamic republic of": "iran",
    "korea": "south korea",
    "korea democratic people's republic of": "north korea",
    "korea republic of": "south korea",
    "lao people's democratic republic": "laos",
    "macedonia": "north macedonia",
    "moldova republic of": "moldova",
    "russian federation": "russia",
    "slovak republic": "slovakia",
    "syrian arab republic": "syria",
    "taiwan province of china": "taiwan",
    "tanzania united republic of": "tanzania",
    "turkey": "turkiye",
    "united states of america": "united states",
    "venezuela bolivarian republic of": "venezuela",
    "viet nam": "vietnam",
}

OECD_MEMBERS = {
    "australia",
    "austria",
    "belgium",
    "canada",
    "chile",
    "colombia",
    "costa rica",
    "czechia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "greece",
    "hungary",
    "iceland",
    "ireland",
    "israel",
    "italy",
    "japan",
    "south korea",
    "latvia",
    "lithuania",
    "luxembourg",
    "mexico",
    "netherlands",
    "new zealand",
    "norway",
    "poland",
    "portugal",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
    "switzerland",
    "turkiye",
    "united kingdom",
    "united states",
}

EURASIA = {
    "armenia",
    "azerbaijan",
    "belarus",
    "georgia",
    "kazakhstan",
    "kyrgyzstan",
    "moldova",
    "russia",
    "tajikistan",
    "turkmenistan",
    "ukraine",
    "uzbekistan",
}


def normalize_country(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return ALIASES.get(text, text)


def is_oecd(country: object) -> bool:
    return normalize_country(country) in OECD_MEMBERS


def irena_cost_region(
    country: object, gem_region: object, gem_subregion: object, technology: str
) -> str:
    name = normalize_country(country)
    if name in {"brazil", "china", "india", "united states"}:
        return {"united states": "United States"}.get(name, name.title())
    if name in EURASIA:
        return "Eurasia"

    region = str(gem_region or "").strip().lower()
    subregion = str(gem_subregion or "").strip().lower()
    if region == "africa":
        return "Africa"
    if region == "europe":
        return "Europe"
    if region == "oceania":
        return "Oceania"
    if region == "asia":
        return "Other Asia"
    if subregion == "northern america":
        return "Other North America"
    if subregion == "latin america and the caribbean":
        if technology == "solar":
            central_america = {
                "belize",
                "costa rica",
                "cuba",
                "dominican republic",
                "el salvador",
                "guatemala",
                "haiti",
                "honduras",
                "jamaica",
                "nicaragua",
                "panama",
                "puerto rico",
                "trinidad and tobago",
            }
            if name in central_america:
                return "Central America and the Caribbean"
        return "Other South America"
    return "Global"
