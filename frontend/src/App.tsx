import { Routing } from "@components/Routing/Routing";
import { createTheme, MantineProvider } from "@mantine/core";
import { BrowserRouter } from "react-router";

import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";

export const App = () => {
	const theme = createTheme({});

	return (
		<MantineProvider theme={theme}>
			<BrowserRouter>
				<Routing />
			</BrowserRouter>
		</MantineProvider>
	);
};
