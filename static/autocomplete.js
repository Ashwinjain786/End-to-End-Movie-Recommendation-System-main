(function () {
    const input = document.getElementById("autoComplete");
    const menu = document.getElementById("suggestions-menu");
    const films = Array.isArray(window.films) ? window.films : [];
    let activeIndex = -1;
    let currentItems = [];

    if (!input || !menu || films.length === 0) {
        return;
    }

    function normalize(value) {
        return String(value || "").trim().toLowerCase();
    }

    function highlight(title, query) {
        const cleanTitle = String(title);
        const index = normalize(cleanTitle).indexOf(normalize(query));
        if (index < 0) {
            return cleanTitle;
        }

        const before = cleanTitle.slice(0, index);
        const match = cleanTitle.slice(index, index + query.length);
        const after = cleanTitle.slice(index + query.length);
        return `${before}<mark>${match}</mark>${after}`;
    }

    function closeMenu() {
        menu.classList.remove("active");
        menu.innerHTML = "";
        activeIndex = -1;
        currentItems = [];
        input.setAttribute("aria-expanded", "false");
    }

    function setActive(index) {
        activeIndex = index;
        Array.from(menu.children).forEach((item, itemIndex) => {
            item.classList.toggle("active", itemIndex === activeIndex);
            item.setAttribute("aria-selected", String(itemIndex === activeIndex));
        });
    }

    function selectItem(index) {
        const item = currentItems[index];
        if (!item) {
            return;
        }
        input.value = item;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        closeMenu();
    }

    function render(query) {
        const search = normalize(query);
        if (search.length < 2) {
            closeMenu();
            return;
        }

        currentItems = films
            .filter(title => normalize(title).includes(search))
            .slice(0, 7);

        if (currentItems.length === 0) {
            menu.innerHTML = '<div class="suggestion-empty">No matching titles</div>';
            menu.classList.add("active");
            input.setAttribute("aria-expanded", "true");
            return;
        }

        menu.innerHTML = currentItems.map((title, index) => (
            `<button type="button" class="suggestion-item" role="option" aria-selected="false" data-index="${index}">
                <span class="fa fa-film" aria-hidden="true"></span>
                <span>${highlight(title, query)}</span>
            </button>`
        )).join("");

        menu.classList.add("active");
        input.setAttribute("aria-expanded", "true");
        setActive(-1);
    }

    input.addEventListener("input", () => {
        render(input.value);
    });

    input.addEventListener("keydown", event => {
        if (!menu.classList.contains("active") || currentItems.length === 0) {
            return;
        }

        if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((activeIndex + 1) % currentItems.length);
        }

        if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive(activeIndex <= 0 ? currentItems.length - 1 : activeIndex - 1);
        }

        if (event.key === "Enter" && activeIndex >= 0) {
            event.preventDefault();
            selectItem(activeIndex);
        }

        if (event.key === "Escape") {
            closeMenu();
        }
    });

    menu.addEventListener("mousedown", event => {
        const item = event.target.closest(".suggestion-item");
        if (!item) {
            return;
        }
        event.preventDefault();
        selectItem(Number(item.dataset.index));
    });

    document.addEventListener("click", event => {
        if (!event.target.closest(".search-control")) {
            closeMenu();
        }
    });
})();
