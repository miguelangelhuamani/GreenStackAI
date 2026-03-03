class RefactoringAgent:
    """
    The main agent class responsible for autonomously refactoring inefficient code.
    Follows a ReAct loop to profile, critique, and edit code.
    """

    def plan_refactor(self, source_code):
        """
        Teammates must implement the Critic logic here.

        This method should:
        1. Analyze the provided source_code.
        2. Consider profiling metrics like time and memory hotspots.
        3. Hypothesize the root cause of inefficiency.
        4. Return a structured string outlining the plan to improve algorithmic complexity.
        """
        return ""

    def execute_cycle(self):
        """
        Teammates must implement the Refiner and the ReAct loop here.

        This method should:
        1. Invoke the Critic using plan_refactor to get a rewrite strategy.
        2. Generate targeted code transformations based on algorithmic principles.
        3. Apply transformations to create a candidate rewrite.
        4. Verify correctness using unit tests.
        5. Re-profile to quantify the percentage runtime speedup and peak RAM reduction.
        6. Loop until performance goals are met or plateaus.
        """
        pass
