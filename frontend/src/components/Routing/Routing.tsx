import { Main } from "@pages/Main/Main";
import { NotFound } from "@pages/NotFound/NotFound";
import { Recognize } from "@pages/Recognize/Recognize";
import { Route, Routes } from "react-router";

export const Routing = () => {
	return (
		<Routes>
			<Route path="/" element={<Main />} />
			<Route path="/recognize/:id" element={<Recognize />} />
			<Route path="*" element={<NotFound />} />
		</Routes>
	);
};
