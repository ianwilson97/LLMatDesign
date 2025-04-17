import os
import ase
import ast
import json
import ase.io
import numpy as np
import random

from llmatdesign.utils import extract_python_code

ask_expert_code_prompt_template = """
I have a material and its band gap value. A band gap is the distance \
between the valence band of electrons and the conduction band, \
representing the minimum energy that is required to excite an electron to the conduction band.

(<chemical_formula>, <band_gap>)

Please propose a modification to the material that results in a band gap of 1.4 eV. \
You can choose one of the four following modifications:
1. exchange: exchange two elements in the material
2. substitute: substitute one element in the material with another
3. remove: remove an element from the material
4. add: add an element to the material

Your output should be a python dictionary of the following the format: {Hypothesis: $HYPOTHESIS, Modification: [$TYPE, $ELEMENT_1, $ELEMENT_2]}. Here are the requirements:
1. $HYPOTHESIS should be your analysis and reason for choosing a modification
2. $TYPE should be the modification type; one of "exchange", "substitute", "remove", "add"
3. $ELEMENT should be the selected element type to be modified. For "exchange" and "substitute", two $ELEMENT placeholders are needed. For "remove" and "add", one $ELEMENT placeholder is needed.\n
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

def format_prompt(suggestions_list, structures_list, properties_list, reflections_list, property_type, target_property):
    past_modifications = get_past_modifications(suggestions_list, structures_list, properties_list, reflections_list)

    prompt = ask_expert_code_prompt_template.replace("<chemical_formula>", structures_list[-1].get_chemical_formula('metal'))
    prompt = prompt.replace("<band_gap>", f"{properties_list[-1]:.2f}")

    if past_modifications is not None:
        prompt += past_modifications

    return prompt

def format_historyless_prompt(suggestions_list, structures_list, properties_list, reflections_list, property_type, target_property):
    # past_modifications = get_past_modifications(suggestions_list, structures_list, properties_list, reflections_list)

    prompt = ask_expert_code_prompt_template.replace("<chemical_formula>", structures_list[-1].get_chemical_formula('metal'))
    prompt = prompt.replace("<band_gap>", f"{properties_list[-1]:.2f}")

    return prompt

def get_reflection_prompt(previous_chemical_formula, current_chemical_formula, modification, target_value, previous_value, current_value):
    base = (
        f"After completing the following modification on the material {previous_chemical_formula}, we obtained {current_chemical_formula} "
        f"the band gap value changed from {previous_value:.2f} eV to {current_value:.2f} eV. "
        f"Please write a post-action reflection on the modification in a short sentence "
        f"on how successful the modification was in achieving the target band gap value of {target_value} eV and why so:\n"
        f"<modification>"
    )

    base = base.replace("<previous_chemical_formula>", previous_chemical_formula)
    base = base.replace("<current_chemical_formula>", current_chemical_formula)
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

            action_str = get_action(agent.llm, prompt)
            print(f"Suggestion: {action_str}; {structures_list[-1].get_chemical_formula('metal')}")
            action = ast.literal_eval(action_str)

            print("Operation: ", action["Modification"])
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
    count = 0
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