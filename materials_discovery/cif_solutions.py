import os
import ase
import ast
import json
import ase.io
import numpy as np
import random
from ase import Atoms

from llmatdesign.utils import extract_python_code

ask_expert_code_prompt_template = """
I have a material and its band gap value. A band gap is the distance \
between the valence band of electrons and the conduction band, \
representing the minimum energy that is required to excite an electron to the conduction band. \
The material is represented by the lattice lengths, lattice angles, followed by \
the atomic species and their fractional coordinates in the unit cell. 

Material:
<material_cif>

Band gap:
<band_gap>

Please propose a modification to the material that results in a band gap of 1.4 eV. \
You can choose one of the four following modifications:
1. exchange: exchange two atoms in the material
2. substitute: substitute one atom in the material with another
3. remove: remove an atom from the material
4. add: add an atom to the material

Your output should be a python dictionary of the following the format: {Hypothesis: $HYPOTHESIS, Modification: [$TYPE, $ATOM_1, $ATOM_2]}. Here are the requirements:
1. $HYPOTHESIS should be your analysis and reason for choosing a modification
2. $TYPE should be the modification type; one of "exchange", "substitute", "remove", "add"
3. $ATOM should be the selected atom to be modified. For "exchange" and "substitute", two $ATOM placeholders are needed. For "remove" and "add", one $ATOM placeholder is needed.
4. $ATOM should be the element name with its index. For example: Na1.
5. For "add", $ATOM index does not need to be specified.
6. For "subsitute", $ATOM_1 needs to be indexed while $ATOM_2 does not need to be indexed.
"""

def get_past_modifications(suggestions_list, structures_list, properties_list, reflections_list):
    history = ""
    if suggestions_list[-1] is None:
        return None
    else:
        history += "You may also want to make use of the past modifications below:\n"

    # enumerate
    for i, (suggestion, structure, property_value, reflection) in enumerate(zip(suggestions_list[1:], structures_list[1:], properties_list[1:], reflections_list[1:])):
        l = f"{i+1}. Modification: {suggestion}. Post-modification reflection: {reflection}.\n"
        history += l

    return history

def struct2cartesian(
    atoms,
    fractional_coors=True
):
    """
    Given the atomic symbols, positions and cell of a structure,
    return a string representation of the structure (CIF).

    Args:
        fractional_coors (bool): Whether to use fractional coordinates or not.
    """
    atomic_symbols = atoms.get_chemical_symbols()
    atom_counter = {k: 1 for k in atomic_symbols}

    lattice_params = atoms.cell.cellpar()
    lengths, angles = lattice_params[:3], lattice_params[3:]
    coors = atoms.get_scaled_positions() if fractional_coors else atoms.get_positions()

    cif_str = " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
            " ".join([str(int(x)) for x in angles]) + "\n"
    
    for t, c in zip(atomic_symbols, coors):
        cif_str += "{0}{1} {2:.2f} {3:.2f} {4:.2f}\n".format(t, atom_counter[t], *c)
        atom_counter[t] += 1
    
    return cif_str.strip()

def format_prompt(suggestions_list, structures_list, properties_list, reflections_list, property_type, target_property):
    past_modifications = get_past_modifications(suggestions_list, structures_list, properties_list, reflections_list)

    prompt = ask_expert_code_prompt_template.replace("<material_cif>", struct2cartesian(structures_list[-1]))
    prompt = prompt.replace("<band_gap>", f"{properties_list[-1]:.2f}")

    if past_modifications is not None:
        prompt += past_modifications

    return prompt

def format_historyless_prompt(suggestions_list, structures_list, properties_list, reflections_list, property_type, target_property):
    # past_modifications = get_past_modifications(suggestions_list, structures_list, properties_list, reflections_list)

    prompt = ask_expert_code_prompt_template.replace("<material_cif>", struct2cartesian(structures_list[-1]))
    prompt = prompt.replace("<band_gap>", f"{properties_list[-1]:.2f}")

    return prompt

def get_reflection_prompt(previous_material_cif, current_material_cif, modification, target_value, previous_value, current_value):
    base = (
        f"After completing the following modification on the material {previous_material_cif}, we obtained {current_material_cif} "
        f"the band gap value changed from {previous_value:.2f} eV to {current_value:.2f} eV. "
        f"Please write a post-action reflection on the modification in a short sentence "
        f"on how successful the modification was in achieving the target band gap value of {target_value} eV and why so:\n"
        f"<modification>"
    )

    base = base.replace("<previous_chemical_formula>", previous_material_cif)
    base = base.replace("<current_chemical_formula>", current_material_cif)
    base = base.replace("<previous_value>", str(previous_value))
    base = base.replace("<current_value>", str(current_value))
    base = base.replace("<target_value>", str(target_value))
    base = base.replace("<modification>", str(modification))

    return base

def solution_base(agent, start_from=1, chemical_formula='SrTiO3', target_value=1.4):
    if start_from <= 1:
        print(f"[Step 1] query Materials Project API to get the structure and band gap of {chemical_formula}")
        if_structure, structure = agent.query_materials_project(chemical_formula, 'structure')
        if_band_gap, band_gap = agent.query_materials_project(chemical_formula, 'band_gap')
        if if_structure and if_band_gap:
            start_from = 3

    if start_from <= 2:
        print(f"[Step 2] generate the crystal structure of {chemical_formula}")
        cif_file_path = agent.generate_crystal(chemical_formula)
        assert os.path.exists(cif_file_path), f'Error in [Step 3]: The returned cif_file_path should exist. {agent.report()}'
        structure = ase.io.read(cif_file_path)

    if start_from <= 3:
        print("[Step 3] ask the expert for suggestions on how to modify the structure")
        number_of_iterations = 50
        suggestions_list = [None]
        structures_list = [structure]
        band_gaps_list = [band_gap]
        reflections_list = [None]

        for i in range(number_of_iterations):
            print(f"Step: {i+1}")

            prompt = format_prompt(
                suggestions_list, 
                structures_list, 
                band_gaps_list,
                reflections_list,
                property_type='band_gap', 
                target_property=target_value
            )

            print(prompt)

            action_str = get_action(agent.llm, prompt)

            print(f"Suggestion: {action_str}; {structures_list[-1].get_chemical_formula('metal')}")

            action = ast.literal_eval(action_str)
            
            new_structure, new_band_gap = agent.perform_cif_modification(structures_list[-1], action["Modification"], calculation_type='band_gap')
            print(f"New band gap: {new_band_gap}; {new_structure.get_chemical_formula('metal')}")

            # get post action reflection
            reflection_prompt = get_reflection_prompt(
                structures_list[-1].get_chemical_formula('metal'),
                new_structure.get_chemical_formula('metal'),
                action_str,
                target_value,
                band_gaps_list[-1],
                new_band_gap
            )

            reflection = get_reflection(agent.llm, reflection_prompt)

            print(f"Reflection: {reflection}")

            suggestions_list.append(action_str)
            structures_list.append(new_structure)
            band_gaps_list.append(new_band_gap)
            reflections_list.append(reflection)

            if agent.is_within_threshold(new_band_gap, target_value):
                print(f"Found a new material with the target band gap: {new_band_gap}")
                return True, suggestions_list, structures_list, band_gaps_list, reflections_list
            
        return False, suggestions_list, structures_list, band_gaps_list, reflections_list
    
def solution_random(agent, start_from=1, chemical_formula='SrTiO3', target_value=1.4):
    pass

def solution_historyless(agent, start_from=1, chemical_formula='SrTiO3', target_value=1.4):
    if start_from <= 1:
        print(f"[Step 1] query Materials Project API to get the structure and band gap of {chemical_formula}")
        if_structure, structure = agent.query_materials_project(chemical_formula, 'structure')
        if_band_gap, band_gap = agent.query_materials_project(chemical_formula, 'band_gap')
        if if_structure and if_band_gap:
            start_from = 3

    if start_from <= 2:
        print(f"[Step 2] generate the crystal structure of {chemical_formula}")
        cif_file_path = agent.generate_crystal(chemical_formula)
        assert os.path.exists(cif_file_path), f'Error in [Step 3]: The returned cif_file_path should exist. {agent.report()}'
        structure = ase.io.read(cif_file_path)

    if start_from <= 3:
        print("[Step 3] ask the expert for suggestions on how to modify the structure")
        number_of_iterations = 50
        suggestions_list = [None]
        structures_list = [structure]
        band_gaps_list = [band_gap]
        reflections_list = [None]

        for i in range(number_of_iterations):
            print(f"Step: {i+1}")

            prompt = format_historyless_prompt(
                suggestions_list, 
                structures_list, 
                band_gaps_list,
                reflections_list,
                property_type='band_gap', 
                target_property=target_value
            )

            action_str = get_action(agent.llm, prompt)

            print(f"Suggestion: {action_str}; {structures_list[-1].get_chemical_formula('metal')}")

            action = ast.literal_eval(action_str)
            
            new_structure, new_band_gap = agent.perform_modification(structures_list[-1], action["Modification"], calculation_type='band_gap')
            print(f"New band gap: {new_band_gap}; {new_structure.get_chemical_formula('metal')}")

            # no post action reflection

            suggestions_list.append(action_str)
            structures_list.append(new_structure)
            band_gaps_list.append(new_band_gap)
            # reflections_list.append(reflection)

            if agent.is_within_threshold(new_band_gap, target_value):
                print(f"Found a new material with the target band gap: {new_band_gap}")
                return True, suggestions_list, structures_list, band_gaps_list, reflections_list
            
        return False, suggestions_list, structures_list, band_gaps_list, reflections_list
    
def get_action(llm, prompt):
    while True:
        llm_response = llm.ask(prompt)
        code = extract_python_code(llm_response)
        code = code.strip()

        if code.startswith("{") and code.endswith("}"):
            return code
        
        count += 1
        if count > 5:
            raise ValueError("Failed to get the action code string.")
        
def get_reflection(llm, prompt):
    return llm.ask(prompt)


if __name__ == "__main__":
    solution_base(None)