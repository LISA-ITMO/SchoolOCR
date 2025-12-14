import { useParams } from "react-router";

export const Recognize = () => {
	const { id } = useParams();

	return <div>recognize with id = {id}</div>;
};
