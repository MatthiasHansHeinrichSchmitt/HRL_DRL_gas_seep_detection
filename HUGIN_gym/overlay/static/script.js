async function fetchBars() {
    try {
        const res = await fetch("/bars");
        const data = await res.json();

        document.getElementById("bar1").style.height = (data[0] * 100) + "%";
        document.getElementById("bar2").style.height = (data[1] * 100) + "%";
        document.getElementById("bar3").style.height = (data[2] * 100) + "%";
    } catch (err) {
        console.error(err);
    }
}

setInterval(fetchBars, 100);