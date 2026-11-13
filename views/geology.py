"""Section 3 - Geological data."""
from ui.interp import page, tab_geology

page.data_page(
    "geology", "4 · Geology",
    "Field mapping and geological maps: lithology map, gold-favourable host rocks, structural measurements "
    "(strike rose, stereonet). A geological map picture or GeoTIFF can be added as a background.",
    tab_geology.render)
