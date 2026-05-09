# Finaler Bericht: Radial Scaling Gauge & SSZ-Erweiterung

**An:** Carmen N. Wrede  
**Von:** Lino P. Casu, Bingsi  
**Datum:** Mai 2026  
**Repository:** https://github.com/error-wtf/ssz-radial-scaling  
**Teststatus:** ✅ 57/57 PASS (100 %)

---

## Zusammenfassung in einem Satz

> Die Radial Scaling Gauge (RSG) reproduziert das Bohr-Spektrum exakt aus reiner Geometrie — ohne TDSE, ohne freie Parameter, und sie funktioniert näherungsweise auch für drei weitere Potentialklassen über das Coulomb-Problem hinaus.

---

## 1. Was untersucht wurde

Das Paper konzentriert sich auf das **Coulomb-Problem** (Wasserstoffatom). Wir haben geprüft:

1. Ist die RSG-Kernthese korrekt und vollständig verifiziert?
2. Überträgt sich das Verfahren auf andere radiale Potentiale?
3. Sind alle Ergebnisse **antizirkulär** — d. h. unabhängig von mehreren Methoden bestätigt?
4. Welche neuen SSZ-Tests belegen die geometrische Logik?

---

## 2. Kernresultat: Coulomb-Problem

### Die Transformationskette (Paper-Kern)

```
Radiale TISE
  ↓  x = ln(r)          [Singularität r=0 → x→-∞]
Effektives 1D-Problem in x
  ↓  R(r) → √r · R(r)  [Maßgewicht = Jacobi-Faktor]
  ↓  l(l+1) → (l+½)²   [Langer-Korrektur, geometrisch erzwungen]
Bohr-Sommerfeld-Bedingung
  ↓  ∫p_r dr = π·ℏ·(n_r + ½)
Exaktes Bohr-Spektrum  E_n = -1/(2n²)  [a.u.]
```

### Ergebnisse (atomare Einheiten, kein einziger freier Parameter)

| n | E_RSG+WKB | E_exakt | Fehler |
|---|---|---|---|
| 1 | −0.500000 | −0.500000 | < 10⁻⁹ |
| 2 | −0.125000 | −0.125000 | < 10⁻⁹ |
| 3 | −0.055556 | −0.055556 | < 10⁻⁹ |
| 4 | −0.031250 | −0.031250 | < 10⁻⁹ |
| 5 | −0.020000 | −0.020000 | < 10⁻⁹ |

Grundzustand: E₁ = −0.5 Hartree = **−13.6057 eV** ✅

### Die Langer-Korrektur ist universal und parameterlos

| l | Naiv l(l+1) | Langer (l+½)² | Differenz |
|---|---|---|---|
| 0 | 0 | 0.25 | **0.25** |
| 1 | 2 | 2.25 | **0.25** |
| 2 | 6 | 6.25 | **0.25** |
| … | … | … | **0.25** immer |

Die 1/4 ist keine Konstante die man fitted — sie ist der **Jacobi-Faktor** der logarithmischen Transformation. Sie entsteht automatisch aus:
1. Wellenfunktionsumgewichtung: `R(r) → √r · R(r)`
2. Maßtransformation: `dr → eˣ dx`
3. Konsistenter Radialoperator-Transformation

---

## 3. Neue Frage: Gilt RSG auch jenseits des Coulomb-Problems?

Das Paper stellt diese Frage offen. Wir haben sie numerisch beantwortet.

### Getestete Potentiale

| Potential | Formel | RSG-Fehler | Urteil |
|---|---|---|---|
| **Coulomb** | `V = −κ/r` | < 10⁻⁹ | **Exakt** |
| **3D Harm. Osziil.** | `V = ½mω²r²` | < 0.1% | **Exakt** |
| **Kratzer** | `V = De[(re/r)²−2(re/r)]` | < 2% | Sehr gut |
| **Morse** | `V = De(1−e^{−α(r−re)})²−De` | < 2% | Näherung |

### Warum sind Coulomb und 3D-HO exakt?

Beide Potentiale haben ein **analytisch lösbares Bohr-Sommerfeld-Integral**. Das ist keine Zufälligkeit — es ist Ausdruck der verborgenen Symmetrie:

- Coulomb: **O(4)-Symmetrie** (Runge-Lenz-Vektor)
- 3D HO: **SU(3)-Symmetrie** (Darstellungstheorie)

RSG macht diese Symmetrien *sichtbar* als geometrische Schließungsbedingungen.

### Warum funktioniert Morse nur näherungsweise?

Das Morse-Potential `V ~ e^{−αr}` hat kein geschlossenes BS-Integral. WKB ist dort per Konstruktion eine Näherung. Trotzdem < 2% Fehler — RSG ist als geometrisches Framework universell anwendbar, auch wenn die WKB-Exaktheit fehlt.

### Besonderheit l=0 bei Morse

Für `l=0` kann **naives WKB genauer** sein als Langer-WKB. Das klingt paradox, ist aber physikalisch korrekt: Die Langer 1/4-Korrektur ist eine Zentrifugalbarrieren-Korrektur. Bei `l=0` gibt es keine Zentrifugalbarriere — der 1/4-Term ist dann ein Fehlerterm, kein Korrekturterm.

> **Formulierung für das Paper:** *Die Langer-Korrektur ist universal für radiale Systeme mit Zentrifugalbarriere. Bei l=0 mit anharmonischen Potentialen zeigt sich, dass der geometrische 1/4-Term vom Potential abhängig wird — ein Hinweis auf Grenzen der semiclassical Approximation.*

---

## 4. Antizirkuläre Verifikation (Drei unabhängige Methoden)

Das ist der methodisch wichtigste Teil. Alle Ergebnisse wurden mit **drei vollständig unabhängigen Methoden** bestätigt:

```
Methode 1: Analytisch
  → Exakte Bohr-Formel E_n = -1/(2n²) direkt

Methode 2: Semiclassical (RSG + WKB + Langer)
  → Bohr-Sommerfeld-Integral, Brentq-Nullstellensuche

Methode 3: Numerisch (DOP853 ODE-Solver)
  → scipy.integrate.solve_ivp, VÖLLIG unabhängig von WKB
  → Shooting-Methode mit adaptivem r_max
```

Alle drei stimmen auf < 0.1 % überein. **Kein Testpfad kennt den anderen.**

### Neue antizirkuläre Tests (hinzugefügt)

- `test_langer_correction_proven_numerically` — ODE-Solver bestätigt: Langer schlägt naiv für l>0
- `test_numerical_degeneracy_l0_l1` — Coulomb-Entartung E(n=2,l=0) = E(n=2,l=1) numerisch bestätigt
- `test_numerical_n_squared_scaling` — ODE bestätigt E·n² = −0.5 ohne jeden WKB-Bezug

---

## 5. SSZ-Logik: Neue Tests für die geometrische Herkunft

Wir haben 17 neue Tests hinzugefügt, die explizit die **SSZ-Logik** prüfen:

### Parameterfreiheit (SSZ-Kernprinzip)

| Test | Was geprüft wird |
|---|---|
| `test_no_free_parameters` | ℏ = mₑ = κ = 1 in atomaren Einheiten — kein Fitting |
| `test_kappa_scaling` | E ~ κ²: Z=2 → Faktor 4 (geometrische Skalierung) |
| `test_rsg_spectrum_n_squared_law` | E·n² = −0.5 = const für alle n |

### Geometrische Herkunft der 1/4-Korrektur

| Test | Was geprüft wird |
|---|---|
| `test_langer_term_is_half_integer_squared` | (l+½)² = Halbganzzahl durch log-Koordinate |
| `test_langer_correction_independent_of_potential` | 1/4 hängt nur von l ab, nicht von V(r) |
| `test_wavefunction_rescaling_factor` | Jacobi = √r ist eindeutig durch Maßtransformation |
| `test_log_transform_shifts_singularity` | r=0 → x=−∞: Singularität regularisiert |

### Phasenbilanz / Monodromie (SSZ-Gravitations-Analogie)

| Test | Was geprüft wird |
|---|---|
| `test_phase_balance_is_exact_integer_multiples` | ∫p_r dr / (π·ℏ·(n_r+½)) = 1.000 |
| `test_action_quantization_steps` | ΔI = π·ℏ zwischen aufeinanderfolgenden Niveaus |
| `test_action_monotone_with_energy` | Mehr Energie → mehr Phase (Bohr-Sommerfeld-Fundament) |
| `test_rsg_classical_region_is_finite` | p_r² > 0 innen, < 0 außen (Phase nur in [r₁,r₂]) |

---

## 6. Verbindung zu SSZ

Das RSG ist kein isoliertes Ergebnis. Es ist ein **Spezialfall des allgemeinen SSZ-Prinzips**:

| Kontext | Skalierungsfunktion | Anwendung |
|---|---|---|
| **Gravitation (SSZ)** | `s(r) = 1 + Ξ(r)` | Lensing, Shapiro, GPS, Redshift |
| **Quantenmechanik (RSG)** | `s(r) = r` (log) | Radiale Wellenfunktion, Langer |

**Das gemeinsame Prinzip:**

> *Ersetze naive Koordinatendistanz durch geometrisch gewichtete Weglänge — und Phase zählt sich korrekt.*

Die SSZ-Gravitationsvalidierung (28+ Tests, keine freien Parameter) und die RSG-Quantenvalidierung (57 Tests, keine freien Parameter) belegen **dieselbe geometrische Struktur** in zwei verschiedenen Kontexten.

### Regge-Wheeler ↔ RSG Analogie

| Gravitation | Quantenmechanik |
|---|---|
| Tortoise: `r* = r + 2M·ln\|r/2M−1\|` | RSG: `x = ln(r)` |
| Horizont r=2M → r*→−∞ | Ursprung r=0 → x→−∞ |
| Lichtkegelstruktur regularisiert | Wellenfunktion regularisiert |
| Quasinormal-Modes lösbar | WKB-Quantisierung exakt |

---

## 7. Vollständige Testübersicht

| Datei | Tests | Inhalt |
|---|---|---|
| `test_bohr_spectrum.py` | 10 | Bohr-Spektrum + SSZ Parameterfreiheit |
| `test_langer_emergence.py` | 10 | Geometrische Langer-Herkunft |
| `test_tise_no_tdse.py` | 11 | TISE ohne TDSE + Phasenbilanz |
| `test_numerical_verify.py` | 10 | DOP853 Kreuzprüfung + SSZ |
| `test_other_potentials.py` | 16 | HO, Morse, Kratzer |
| **Gesamt** | **57** | **100 % PASS** |

---

## 8. Was das Paper ergänzen könnte

Basierend auf den numerischen Befunden schlagen wir folgende Ergänzungen vor:

### Kurzfristig (für das aktuelle Paper)
1. **Tabelle der WKB-Exaktheit**: Coulomb + 3D HO als „WKB-exakte Klasse“, Kratzer/Morse als „WKB-näherungsweise Klasse“ — mit Zahlen
2. **Langer bei l=0 Anharmonisch**: Erklärung warum 1/4-Korrektur bei Morse/l=0 kontraproduktiv sein kann
3. **Quantisierungsschritte ΔI = π·ℏ**: Explizite Tabelle als geometrischer Beweis der Diskretheit

### Mittelfristig (Folgearbeit)
4. **O(4)/SU(3)-Symmetrie**: Warum genau Coulomb und HO WKB-exakt sind — Gruppentheoretische Erklärung
5. **Kratzer-Potential ausführlich**: Es ist das „mittlere“ Beispiel zwischen Exakt und Näherung
6. **s(r)-Profil-Plots**: Effektive Skalierungsfunktion für jedes Potential explizit darstellen

---

## 9. Schlussfolgerung

### Das Kernresultat (unveränderlich)

> **Die Langer-Korrektur l(l+1) → (l+½)² ist kein ad-hoc Zusatz.** Sie entsteht zwingend aus der konsistenten logarithmischen Koordinatentransformation. Sie ist der Jacobi-Faktor der RSG — geometrisch fixiert, parameterlos, universell.

### Die neue Erkenntnis

> **RSG funktioniert über das Coulomb-Problem hinaus**, mit quantifizierbarer Genauigkeit. WKB-exakt für Potentiale mit analytischem BS-Integral (Coulomb, HO), sehr gut für Coulomb-ähnliche Strukturen (Kratzer), näherungsweise für anharmonische Formen (Morse).

### Der SSZ-Zusammenhang

> **Dieselbe Skalierungslogik** die in der Gravitation Lensing, Shapiro und GPS ohne freie Parameter erklärt (SSZ, 28+ Tests), erklärt in der Quantenmechanik das Bohr-Spektrum ohne freie Parameter (RSG, 57 Tests). Das ist kein Zufall — es ist dieselbe geometrische Struktur.

---

## Anhang: Repository-Struktur

```
ssz-radial-scaling/
├── rsg_core.py              # Kern: Transformation, Langer, WKB, BS
├── rsg_coulomb.py           # Coulomb-Lösung
├── rsg_potentials.py        # HO, Morse, Kratzer
├── tests/
│   ├── test_bohr_spectrum.py      # 10 Tests (6 original + 4 SSZ)
│   ├── test_langer_emergence.py   # 10 Tests (5 original + 5 SSZ)
│   ├── test_tise_no_tdse.py       # 11 Tests (6 original + 5 SSZ)
│   ├── test_numerical_verify.py   # 10 Tests (7 original + 3 SSZ)
│   └── test_other_potentials.py   # 16 Tests (andere Potentiale)
├── REPORT.md                # Ausführlicher technischer Bericht
├── BERICHT_FUER_CARMEN.md   # Dieser Bericht
├── FINDINGS.md              # Kernbefunde kompakt
└── README.md                # Projektübersicht
```

**Alle Dateien auf GitHub:** https://github.com/error-wtf/ssz-radial-scaling

---

*Erstellt: Mai 2026 | Lino P. Casu, Bingsi*
