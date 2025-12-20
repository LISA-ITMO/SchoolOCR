export const downloadJson = (data: unknown, filename = "data.json") => {
	const json = JSON.stringify(data, null, 2);
	const blob = new Blob([json], { type: "application/json;charset=utf-8" });

	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();

	a.remove();
	URL.revokeObjectURL(url);
};
