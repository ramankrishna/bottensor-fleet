from fleet import Graph, Agent

printer = Agent(
    name="printer",
    goal="Use the pretty_print tool to print the user's message exactly as they describe.",
    model="anthropic/claude-sonnet-4-6",
    tools=["pretty_print"],
)

graph = (
    Graph("pretty_print_demo")
    .add_node("printer", printer.step)
    .set_entry("printer")
    .set_exit("printer")
    .compile()
)
