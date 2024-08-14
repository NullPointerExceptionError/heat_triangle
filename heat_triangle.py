import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize # für ColorMap
import numpy as np
import re # für regex
from collections import defaultdict # für dict, das default Werte hat (leeres dict wirft keinen Fehler)

# Funktion zum Umrechnen von baryzentrischen Koordinaten zu kartesischen
def barycentric_to_cartesian(p:tuple, vertices:np.array) -> np.ndarray:
    """
    converts a point from barycentric to cartesian coordinates
    :param p: point in barycentric coordinates
    :param vertices: vertices of the triangle
    :return: point in cartesian coordinates
    """
    return p[0] * vertices[0] + p[1] * vertices[1] + p[2] * vertices[2] # Brechnung: jeweils Gewicht*Eckkoordinate

def extract_molecule_name(molecule_energies:str) -> dict:
    """
    extracts the molecule name and its energy from the output of the molecule_energies.py
    :param molecule_energies: output of the molecule_energies.py
    :return: dictionary containing the molecule name and its energy
    """
    molecule_dict = {}
    molecule_energies = molecule_energies.split("\n") # in Array umwandeln
    molecule_energies = [line for line in molecule_energies if line] # leere Zeilen löschen
    energies_regex = r"ΔE_d\(([^)]+)\) = (-?\d+\.\d+) eV"
    for entry in molecule_energies:
        match = re.search(energies_regex, entry) # 
        if match:
            molecule_name = match.group(1)  # Extrahiere den Molekülnamen
            value = float(match.group(2))  # Extrahiere den Wert und konvertiere ihn in eine float
            molecule_name = molecule_name.split('_')[0] # Molekülnamen nach _ abschneiden
            molecule_dict[molecule_name] = value
    return molecule_dict

def extract_elements(formula:str, elements:list) -> dict:
    """
    extracts the elements and their counts from a chemical formula
    :param formula: chemical formula
    :return: dictionary containing the elements and their counts
    """
    element_regex = '|'.join(elements) # regex für Elemente
    moleculename_regex = rf'({element_regex})(\d*\.?\d*x?)' # regex für Elemente mit Zahlen bzw x

    # moleculename_regex = r'([A-Z][a-z]*)([\d.]*x?)' # regex für Elemente und Zahlen
    matches = re.findall(moleculename_regex, formula) # alle Elemente und Zahlen finden
    
    element_counts = defaultdict(float) # default dict, da normales dict Fehler wirft, wenn Element nicht existiert
    
    for (element, count) in matches:
        if count == '': # wenn keine Zahl hinter dem Element steht, dann ist die Anzahl 1
            count = float(1)
        elif 'x' in count:
            count = float(count.replace('x', '')) * 0.5 if count != 'x' else 0.5
        else:
            count = float(count)
        element_counts[element] += count # Element und Anzahl in dict speichern

    # Fehlende Elemente mit Anzahl 0 hinzufügen
    for element in elements:
        if element not in element_counts:
            element_counts[element] = float(0)

    return dict(element_counts)

# orthogonaler Abstand Punkt zu Ebene
# def signed_distance_to_plane(point:np.array, plane_point:np.array, normal_vector:np.array) -> float:
#     """
#     Berechnet den vorzeichenbehafteten orthogonalen Abstand eines Punktes von einer Ebene.
    
#     :param point: Der Punkt, dessen Abstand zur Ebene berechnet werden soll (als Vektor).
#     :param plane_point: Ein Punkt auf der Ebene (als Vektor).
#     :param normal_vector: Der Normalenvektor der Ebene.
#     :return: Der Vorzeichenbehaftete Abstand des Punktes von der Ebene.
#     """
#     point_to_plane = point - plane_point # Berechne den Vektor vom Punkt auf der Ebene zum gegebenen Punkt    
#     distance = np.dot(normal_vector, point_to_plane) / np.linalg.norm(normal_vector) # Berechne den Abstand (Skalarprodukt von Normalenvektor und Punktvektor)
    
#     return distance

def signed_distance_to_plane(point:np.array, normal_vector:np.array, d_plane:float) -> float:
    """
    Berechnet den vorzeichenbehafteten Abstand eines Punktes von einer Ebene (auf z-Achse gerade nach oben/unten).
    
    :param point: Der Punkt, dessen Abstand zur Ebene berechnet werden soll (als Vektor).
    :param normal_vector: Der Normalenvektor der Ebene.
    :param d_plane: Der Abstand der Ebene zum Ursprung (rechte Seite der Ebenengleichung in Koordinatenform).
    :return: Der Vorzeichenbehaftete Abstand des Punktes von der Ebene.
    """
    # x3-Koordinate des Punktes auf der Ebene berechnen:
    # x3 = (d - n1*x1 - n2*x2) / n3
    x3_plane = (d_plane - normal_vector[0] * point[0] - normal_vector[1] * point[1]) / normal_vector[2]

    # vorzeichenbehafteter Abstand des Punktes zur Ebene
    return point[2] - x3_plane # Punkt - Ebene (bzgl. x3-Koordinate)


def normalize_energy(energy:float, max_green:float, max_red:float):
    """
    normalizes the energy-distances for color-calculating
    :param energy: energy value to normalize
    :param max_green: min-energy
    :param max_red: max-energy
    :return: value between 0 and 1 to insert energy in colormap
    """
    # Wenn max_red == max_green, alle Werte sind gleich
    if max_red == max_green:
        return 0.5 # weiß
    if energy < 0:
        # Normalisierung für negative Werte
        normalized = (energy - max_green) / (0 - max_green)
        return np.clip(normalized * 0.5, 0, 0.5) # Skallierung auf [0,0.5] -> grün bis weiß
    else:
        # Normalisierung für positive Werte
        normalized = (energy - 0) / (max_red - 0)
        return np.clip(0.5 + normalized * 0.5, 0.5, 1) # Skallierung auf [0.5,1] -> weiß bis rot

# #############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################
# aus der convex_hull.py kopieren (darf mehr Daten als nötig enthalten, hauptsache Energiewerte stimmen mit molecule_energies überein)
energies = np.array([

    [1.0, 0.0, -9.64580],
    [0.0, 0.0, -1.26808],
    [0.0, 1.0, -4.66455],
    [0.4, 0.2, -5.17630],
    [0.2, 0.6, -4.90325],
    [0.5, 0.0, -5.12201],
    [0.5, 0.0, -5.12210],

])

# Ausgabe der energies.py hier reinkopieren (doppelte Moleküle löschen)
molecule_energies = """

ΔE_d(C) = -9.64580 eV
ΔE_d(Na) = -1.26808 eV
ΔE_d(Pd) = -4.66455 eV
ΔE_d(Na2PdC2) = -5.17630 eV
ΔE_d(NaPd3Cx) = -4.90325 eV
ΔE_d(Na2C2_I41/acd) = -5.12201 eV

"""

# Das x für Cx
x = 0.5

# Elemente (genau 3) -> andere Teile im Code müssen bei anderen Elementen angepasst werden (Beschriftung der Eckpunkte & Reihenfolge wichtig für baryzentrischen Punkt)
elements = ["C", "Na", "Pd"]
# #############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################

molecule_names = extract_molecule_name(molecule_energies)

# Molekülnamen mit [x,y,energie] zusammenbringen
data = {}
for molecule_name in molecule_names:
    for energy in energies:
        if energy[2] == molecule_names[molecule_name]: # Energie aus energies mit Energie aus energies.py vergleichen
            data[molecule_name] = {"energy": energy} # Molekülnamen als Key, Energie als Value
            data[molecule_name]["composition"] = extract_elements(molecule_name, elements) # Elemente und Anzahl in dict speichern



# Ebene definieren
oa_vec = data[elements[0]]["energy"] # erster Eckpunkt der Ebene
ob_vec = data[elements[1]]["energy"] # zweiter Eckpunkt der Ebene
oc_vec = data[elements[2]]["energy"] # dritter Eckpunkt der Ebene

ab_vec = ob_vec - oa_vec # Vektor von A nach B
ac_vec = oc_vec - oa_vec # Vektor von A nach C

if np.cross(ab_vec, ac_vec)[2] < 0: # Wenn der Normalenvektor nach unten zeigt, Vektor negieren
    normal_vec = -np.cross(ab_vec, ac_vec) # Normalenvektor der Ebene
else:
    normal_vec = np.cross(ab_vec, ac_vec) # Normalenvektor der Ebene. Negativ, damit er nach oben zeigt
d_plane = oa_vec @ normal_vec # Abstand der Ebene zum Ursprung (@ = Standard-Skalarprodukt)


# Dreieckskoordinaten
triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])


# Plot erstellen
fig, ax = plt.subplots()
triangle = plt.Polygon(triangle_vertices, edgecolor='k', facecolor='none') # Dreieck: schwarze Linien, keine Füllung
ax.add_patch(triangle)

# Eckpunkte beschriften
labels = ['C', 'Na', 'Pd']
for i, (vertex, label) in enumerate(zip(triangle_vertices, labels)):
    # Position des Textes leicht verschieben, um Überlappungen zu vermeiden
    if label == 'Pd': # oben
        ax.text(vertex[0], vertex[1] + 0.05, label, fontsize=12, ha='center', va='center')
    elif label == 'C': # unten links
        ax.text(vertex[0] - 0.05, vertex[1] - 0.05, label, fontsize=12, ha='center', va='center')
    elif label == 'Na': # unten rechts
        ax.text(vertex[0] + 0.05, vertex[1] - 0.05, label, fontsize=12, ha='center', va='center')

# berechnet maximale und minimale Energie (für Farbskala)
max_red = 0 # falls kein Punkt oberhalb der Ebene ist
max_green = 0 # falls kein Punkt unterhalb der Ebene ist
for molecule in data:
    signed_distance = round(signed_distance_to_plane(data[molecule]["energy"], normal_vec, d_plane), 5) # Abstand Punkt <-> Ebene
    if signed_distance > max_red: # neuer maximal positiver Wert
        max_red = signed_distance
    elif signed_distance < max_green: # neuer maximal negativer Wert
        max_green = signed_distance

# Farbskala symmetrisch machen
if max_red < -max_green: # negativer Wert ist größer
    max_red = -max_green
elif -max_green < max_red:
    max_green = -max_red

# Sonderfälle (Skala nur negativ oder positiv)
if max_red == 0 and max_green == 0: # nur gleiche Werte (Standardintervall [-1,1] als Farbskala)
    max_red = 1
    max_green = -1
elif max_red == 0: # nur negative Werte (Skala spiegeln)
    max_red = -max_green
elif max_green == 0: # nur positive Werte (Skala spiegeln)
    max_green = -max_red


# Definiert eine benutzerdefinierte Colormap von Grün über Weiß nach Rot
colors = [(0, 'green'), (0.5, 'white'), (1, 'red')]  # Farbverlauf von grün über weiß nach rot
cmap_name = 'green_white_red'
n_bins = 100000  # Anzahl der Farben im Verlauf
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

for molecule in data:
    total_amount_elements = sum(data[molecule]["composition"].values()) # Summe der Elemente
    barycentric_point = [data[molecule]["composition"][element] / total_amount_elements for element in elements] # Anteil der Elemente als baryzentrischer Punkt, REIHENFOLGE IN elements WICHTIG!
    cartesian_point = barycentric_to_cartesian(barycentric_point, triangle_vertices) # Punkt in kartesischen Koordinaten

    energy = round(signed_distance_to_plane(data[molecule]["energy"], normal_vec, d_plane), 5)
    normalized_energy = normalize_energy(energy, max_green, max_red)
    color = cm(normalized_energy)

    if energy == 0:
        ax.scatter(cartesian_point[0], cartesian_point[1], color=color, s=40, edgecolor="black", linewidth=1) # Umrandung falls weiß
    else:
        ax.scatter(cartesian_point[0], cartesian_point[1], color=color, s=40)  # zeichnet Punkt mit Farbe color und Größe 40


# Achsenlimits setzen
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# Achsen entfernen
ax.axis('off')

# Verhältnis der Achsen einstellen
plt.gca().set_aspect('equal', adjustable='box') # quadratisch (Seitenverhältnis 1:1)

#ax.set_title('') # Diagrammtitel

# Erstelle und füge die Colorbar hinzu
norm = Normalize(vmin=max_green, vmax=max_red)
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])  # Setzt die Daten für die Colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Energy')
cbar.set_ticks([max_green, 0, max_red])  # Setze Ticks bei max_green, 0 und max_red
cbar.set_ticklabels([f'{max_green:.1f}', '0', f'{max_red:.1f}'])  # Setze die Label für diese Ticks

#plt.savefig('heat_triangle.png', dpi=300, bbox_inches='tight')  # Speichert das Bild als PNG-Datei
plt.savefig('plot_image.svg', format='svg', bbox_inches='tight')  # Speichert das Bild als SVG-Datei

plt.show()
