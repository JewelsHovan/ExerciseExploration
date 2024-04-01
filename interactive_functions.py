import ipywidgets as widgets
from IPython.display import display

class RHRCalculator(widgets.VBox):
    def __init__(self):
        super().__init__()

        self.age = None
        self.gender = None
        self.rhr = None

        # Define the health conditions and corresponding RHR values by age groups as per the tables
        self.health_conditions_men = {
            'Athlete': [52, 50, 53, 54, 56, 55],
            'Excellent': [61, 61, 62, 63, 61, 61],
            'Good': [65, 65, 66, 67, 67, 65],
            'Above Average': [69, 70, 70, 71, 71, 69],
            'Average': [73, 74, 75, 76, 75, 73],
            'Below Average': [81, 81, 82, 83, 81, 79],
            'Poor': [82, 82, 83, 84, 82, 80]
        }

        self.health_conditions_women = {
            'Athlete': [48, 46, 49, 54, 55, 55],
            'Excellent': [65, 64, 64, 65, 64, 64],
            'Good': [69, 68, 69, 69, 68, 68],
            'Above Average': [73, 72, 73, 73, 73, 72],
            'Average': [78, 76, 78, 77, 77, 76],
            'Below Average': [84, 82, 84, 83, 83, 84],
            'Poor': [85, 83, 85, 84, 84, 84]
        }

        # Create the gender toggle buttons
        gender_toggle = widgets.ToggleButtons(
            options=['Men', 'Women'],
            description='Gender:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Calculate for men', 'Calculate for women'],
        )

        # Styling attributes
        style = {'description_width': 'initial'}
        layout = widgets.Layout(width='auto', display='flex')

        # Create widgets with style and layout
        age_input = widgets.IntText(value=40, description='Enter your age:', style=style, layout=layout)
        health_dropdown = widgets.Dropdown(
            options=list(self.health_conditions_men.keys()),
            value='Average',
            description='Select your health condition:',
            style=style,
            layout=layout
        )
        calculate_button = widgets.Button(
            description="Calculate RHR",
            layout=layout,
            button_style='success'  # Use a predefined styling for the button
        )
        output_label = widgets.Label(layout=layout)  # Label to display the result

        # Function to calculate and display the upper bound RHR
        def calculate_display_rhr(age, condition, gender):
            # Select the correct health conditions table based on gender
            if gender == 'Men':
                health_conditions = self.health_conditions_men
            else:
                health_conditions = self.health_conditions_women

            # Determine the age index
            if age < 25:
                age_index = 0
            elif age <= 35:
                age_index = 1
            elif age <= 45:
                age_index = 2
            elif age <= 55:
                age_index = 3
            elif age <= 65:
                age_index = 4
            else:
                age_index = 5

            # Get the RHR values based on health condition and age index
            rhr_values = health_conditions[condition]
            upper_bound_rhr = rhr_values[age_index]
            return upper_bound_rhr

        # Function to handle the button click event
        def on_calculate_button_clicked(b):
            # Get the current values from the widgets
            self.age = age_input.value
            self.gender = gender_toggle.value
            health_condition = health_dropdown.value

            # Calculate the RHR using the selected age, gender, and health condition
            self.rhr = calculate_display_rhr(self.age, health_condition, self.gender)

            # Update the label with the result
            output_label.value = f"The expected resting heart rate (RHR) for {self.gender.lower()}: {self.rhr} bpm"

        # Bind the button click to the event function
        calculate_button.on_click(on_calculate_button_clicked)

        # Set up the interactive output for live update as the user changes the inputs
        interactive_output = widgets.interactive_output(
            calculate_display_rhr,
            {'age': age_input, 'condition': health_dropdown, 'gender': gender_toggle}
        )

        # Layout to organize widgets vertically
        box_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='50%')
        box = widgets.Box(children=[gender_toggle, age_input, health_dropdown, calculate_button, output_label], layout=box_layout)

        # Add the box to the VBox
        self.children = [box, interactive_output]

    def get_values(self):
        return self.age, self.gender, self.rhr

def rhr_calculator():
    calculator = RHRCalculator()
    display(calculator)
    return calculator