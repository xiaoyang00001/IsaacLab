// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * Standalone version switcher for Isaac Lab documentation.
 *
 * Injected into every version's HTML during deploy. Fetches versions.json
 * and renders a floating dropdown. Skips rendering if the pydata-sphinx-theme
 * built-in switcher is already present on the page.
 */
(function () {
  "use strict";

  // Skip if pydata-sphinx-theme's built-in switcher is present.
  if (document.querySelector(".version-switcher__container")) {
    return;
  }

  // Resolve the base URL for the docs site.
  // Works for both /IsaacLab/main/... and custom domain setups.
  var pathParts = window.location.pathname.split("/").filter(Boolean);
  // Expect at least /<project>/<version>/...
  var basePath = "/" + pathParts[0] + "/";
  var currentSlug = pathParts[1] || "";
  // The relative page path within the version (e.g. "api/index.html").
  var pagePath = pathParts.slice(2).join("/") || "index.html";

  var JSON_URL = basePath + "versions.json";

  function createSwitcher(versions) {
    // Container
    var container = document.createElement("div");
    container.id = "isaaclab-version-switcher";

    // Label
    var label = document.createElement("label");
    label.textContent = "Version: ";
    label.setAttribute("for", "isaaclab-version-select");
    container.appendChild(label);

    // Select
    var select = document.createElement("select");
    select.id = "isaaclab-version-select";
    for (var i = 0; i < versions.length; i++) {
      var v = versions[i];
      var option = document.createElement("option");
      option.value = v.url;
      option.textContent = v.name;
      if (v.version === currentSlug) {
        option.selected = true;
      }
      select.appendChild(option);
    }
    select.addEventListener("change", function () {
      var targetBase = select.value;
      // Try to navigate to the same page under the new version.
      var targetUrl = targetBase + pagePath;
      // Use a HEAD request to check if the page exists; fall back to index.
      var xhr = new XMLHttpRequest();
      xhr.open("HEAD", targetUrl, true);
      xhr.onload = function () {
        window.location.href = xhr.status < 400 ? targetUrl : targetBase;
      };
      xhr.onerror = function () {
        window.location.href = targetBase;
      };
      xhr.send();
    });
    container.appendChild(select);

    // Styles
    var style = document.createElement("style");
    style.textContent =
      "#isaaclab-version-switcher {" +
      "  position: fixed; top: 10px; right: 10px; z-index: 9999;" +
      "  background: #2b2b2b; color: #e0e0e0; padding: 6px 12px;" +
      "  border-radius: 6px; font-family: sans-serif; font-size: 13px;" +
      "  box-shadow: 0 2px 8px rgba(0,0,0,0.3);" +
      "}" +
      "#isaaclab-version-switcher label {" +
      "  margin-right: 4px; font-weight: bold;" +
      "}" +
      "#isaaclab-version-switcher select {" +
      "  background: #3c3c3c; color: #e0e0e0; border: 1px solid #555;" +
      "  border-radius: 4px; padding: 2px 6px; font-size: 13px;" +
      "}" +
      "@media (prefers-color-scheme: light) {" +
      "  #isaaclab-version-switcher { background: #f5f5f5; color: #333; }" +
      "  #isaaclab-version-switcher select { background: #fff; color: #333; border-color: #ccc; }" +
      "}";
    document.head.appendChild(style);
    document.body.appendChild(container);
  }

  // Fetch versions.json and build the switcher.
  var xhr = new XMLHttpRequest();
  xhr.open("GET", JSON_URL, true);
  xhr.onload = function () {
    if (xhr.status === 200) {
      try {
        var versions = JSON.parse(xhr.responseText);
        if (!Array.isArray(versions)) {
          console.warn("[version-switcher] versions.json is not an array");
          return;
        }
        createSwitcher(versions);
      } catch (e) {
        console.warn("[version-switcher] Failed to parse versions.json:", e.message);
      }
    }
  };
  xhr.onerror = function () {
    console.warn("[version-switcher] Network error fetching", JSON_URL);
  };
  xhr.send();
})();
