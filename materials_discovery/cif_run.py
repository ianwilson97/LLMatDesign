import os
import sys
import ase
import ase.io
import argparse
from datetime import datetime

from llmatdesign.modules.llms import AskLLM
from llmatdesign.utils import *
from llmatdesign.core.agent import Agent

from cif_solutions import *

def main(args):
    api_key = ""
    openai_organization = ""
    llm = AskLLM(args.llm_model, api_key=api_key, openai_organization=openai_organization)

    agent = Agent(
        llm,
        save_path=args.save_path,
        forcefield_config_path=args.forcefield_config_path,
        bandgap_config_path=args.bandgap_config_path,
        formation_energy_config_path=args.formation_energy_config_path
    )

    if args.solution_type == 'base':
        solution = solution_base
    elif args.solution_type == 'random':
        solution = solution_random
    elif args.solution_type == 'historyless':
        solution = solution_historyless
    else:
        raise NotImplementedError
    
    # Get the current date and time
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_save_path = f"./outputs/{args.chemical_formula}/{args.llm_model}/{args.solution_type}/{date_time_str}/"
    os.makedirs(output_save_path, exist_ok=True)

    success_count = 0
    failure_count = 0

    print(f"Starting run...")

    while True:
        if success_count >= args.success_count:
            print(f"Success count reached: {success_count}")
            break

        if failure_count >= args.failure_count:
            print(f"Failure count reached: {failure_count}")
            break

        try:
            success, suggestions_list, structures_list, band_gaps_list, reflections_list = solution(
                agent, 
                start_from=1, 
                chemical_formula=args.chemical_formula, 
                target_value=args.target_value, 
            )

            success_count += 1

            # save the results
            # create a folder for each solution
            os.makedirs(f"{output_save_path}run_{success_count}", exist_ok=True)

            # save structures
            for i, structure in enumerate(structures_list):
                ase.io.write(f"{output_save_path}run_{success_count}/structure_{i+1}.cif", structure)
            
            # save suggestions as a text file
            with open(f"{output_save_path}run_{success_count}/suggestions.txt", "w") as f:
                for suggestion in suggestions_list:
                    f.write(f"{suggestion}\n")

            # save the band gap values as text file
            with open(f"{output_save_path}run_{success_count}/band_gaps.txt", "w") as f:
                for band_gap in band_gaps_list:
                    f.write(f"{band_gap}\n")

            # save the reflections as text file
            with open(f"{output_save_path}run_{success_count}/reflections.txt", "w") as f:
                for reflections in reflections_list:
                    f.write(f"{reflections}\n")

        except Exception as e:
            print(f"Exception: {e}")
            failure_count += 1
            continue
    
    # create a file to indicate that run has completed
    with open(f"{output_save_path}run_complete.txt", "w") as f:
        f.write(f"Run complete! Success count: {success_count}/{args.success_count}")

    print("Run complete!")
    print(f"Success count: {success_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--api_key", type=str, default=None, help="LLM api key")
    parser.add_argument("--save_path", type=str, default="./outputs/cifs/", help="The path to save the CIF files")
    parser.add_argument("--forcefield_config_path", type=str, default="../checkpoints/matdeeplearn/force_field/config.yml", help="The path to the force field config file")
    parser.add_argument("--bandgap_config_path", type=str, default="../checkpoints/matdeeplearn/band_gap/config.yml", help="The path to the band gap config file")
    parser.add_argument("--formation_energy_config_path", type=str, default="../checkpoints/matdeeplearn/formation_energy/config.yml", help="The path to the formation energy config file")
    parser.add_argument("--solution_type", type=str, default="base", help="The type of solution to run")
    parser.add_argument("--chemical_formula", type=str, default="SrTiO3", help="The chemical formula of the starting material")
    parser.add_argument("--target_value", type=float, default=1.4, help="The target value of the property to be optimized")
    parser.add_argument("--success_count", type=int, default=30, help="The number of successful runs to perform")
    parser.add_argument("--failure_count", type=int, default=100, help="The number of failed runs to allow")
    args = parser.parse_args()

    main(args)