import { Routing } from "@components/Routing/Routing";
import { BrowserRouter } from "react-router";

export const App = () => {
	return (
		<BrowserRouter>
			<Routing />
		</BrowserRouter>
	);
};
