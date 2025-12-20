import { Routing } from "@components/Routing/Routing";
import { createTheme, MantineProvider } from "@mantine/core";
import { BrowserRouter } from "react-router";

import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@mantine/notifications/styles.css";
import { Notifications } from "@mantine/notifications";

export const App = () => {
	const theme = createTheme({});

	return (
		<MantineProvider theme={theme}>
			<Notifications />
			<BrowserRouter>
				<Routing />
			</BrowserRouter>
		</MantineProvider>
	);
};
