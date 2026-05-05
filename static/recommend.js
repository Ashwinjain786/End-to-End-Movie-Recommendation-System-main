(function () {
  const form = document.getElementById("movie-form");
  const input = document.getElementById("autoComplete");
  const button = document.querySelector(".movie-button");
  const results = document.getElementById("results");
  const fail = document.querySelector(".fail");
  const loader = document.getElementById("loader");

  if (!form || !input || !button || !results) {
    return;
  }

  function setLoading(isLoading) {
    button.disabled = isLoading || input.value.trim() === "";
    button.classList.toggle("is-loading", isLoading);
    if (loader) {
      loader.classList.toggle("active", isLoading);
      loader.setAttribute("aria-hidden", String(!isLoading));
    }
  }

  function setMessage(message) {
    if (!fail) {
      return;
    }
    fail.textContent = message || "";
    fail.classList.toggle("active", Boolean(message));
  }

  function setResultHtml(html) {
    results.innerHTML = html;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function loadMovie(title) {
    const cleanTitle = String(title || "").trim();
    if (!cleanTitle) {
      setMessage("Enter a movie title.");
      return;
    }

    setMessage("");
    setLoading(true);

    try {
      const body = new URLSearchParams({ title: cleanTitle });
      const response = await fetch("/movie", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        body,
      });
      const html = await response.text();
      setResultHtml(html);

      if (!response.ok) {
        setMessage("Movie details could not be loaded.");
      }
    } catch (error) {
      setResultHtml(`
        <section class="empty-state error-state">
          <span class="fa fa-exclamation-circle" aria-hidden="true"></span>
          <h2>Unable to load movie details.</h2>
        </section>
      `);
      setMessage("Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  input.addEventListener("input", function () {
    button.disabled = input.value.trim() === "";
    setMessage("");
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    loadMovie(input.value);
  });

  document.querySelectorAll(".quick-picks button").forEach(function (quickPick) {
    quickPick.addEventListener("click", function () {
      input.value = quickPick.dataset.title || "";
      input.dispatchEvent(new Event("input", { bubbles: true }));
      loadMovie(input.value);
    });
  });

  window.recommendcard = function (element) {
    const title = element.dataset.title || element.getAttribute("title") || "";
    input.value = title;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    loadMovie(title);
  };
})();
