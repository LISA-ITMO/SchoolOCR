export const loadData = async <T>(path: string, params?: RequestInit) => {
	try {
		const baseUrl = "http://localhost:8000";
		const url = `${baseUrl}${path}`;
		const request = await fetch(url, params);
		if (!request.ok) {
			throw request;
		}

		const json = await request.json();

		return json as Promise<T>;
	} catch (err) {
		console.error(`error when requesting ${path}: ${err}`);
		throw err;
	}
};
