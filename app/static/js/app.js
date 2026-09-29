/**
 * app.js - Client-side interaction logic for MovieLens Cosine Recommender.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Tự động nhận diện port: nếu chạy trên port 8000 thì dùng relative path, nếu mở bằng Live Server (5500) thì gọi sang 8000
  const API_BASE = (window.location.port === "8000" || window.location.port === "") ? "" : "http://127.0.0.1:8000";

  // Elements
  const navButtons = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const searchInput = document.getElementById("movie-search-input");
  const autocompleteList = document.getElementById("autocomplete-list");
  const searchSpinner = document.getElementById("search-spinner");
  const genreFilter = document.getElementById("filter-genre");
  const kSelect = document.getElementById("select-k");
  const btnRecommend = document.getElementById("btn-search-recommend");

  const seedCard = document.getElementById("seed-movie-card");
  const seedTitle = document.getElementById("seed-title");
  const seedGenres = document.getElementById("seed-genres");
  const seedRatings = document.getElementById("seed-ratings");
  const seedAvgRating = document.getElementById("seed-avg-rating");
  const seedWarningBox = document.getElementById("seed-warning-box");
  const seedWarningMsg = document.getElementById("seed-warning-msg");

  const recsGrid = document.getElementById("recommendations-grid");
  const recsCountTitle = document.getElementById("results-count-title");
  const queryMetaInfo = document.getElementById("query-meta-info");

  const failureTableBody = document.getElementById("failure-table-body");
  const btnReloadFailures = document.getElementById("btn-reload-failures");

  let selectedMovieId = null;
  let searchDebounceTimeout = null;

  // Nạp danh sách thể loại nếu mở qua Live Server (không qua Jinja template)
  if (genreFilter && genreFilter.options.length <= 1) {
    const defaultGenres = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"];
    defaultGenres.forEach(g => {
      const opt = document.createElement("option");
      opt.value = g;
      opt.textContent = g;
      genreFilter.appendChild(opt);
    });
  }

  // ==================== TAB NAVIGATION ====================
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetTabId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (targetTabId === "tab-dashboard") {
        loadFailureCases();
      }
    });
  });

  // ==================== AUTOCOMPLETE SEARCH ====================
  searchInput.addEventListener("input", (e) => {
    const val = e.target.value.trim();
    clearTimeout(searchDebounceTimeout);

    if (val.length < 2) {
      autocompleteList.classList.add("hidden");
      autocompleteList.innerHTML = "";
      return;
    }

    searchDebounceTimeout = setTimeout(async () => {
      searchSpinner.classList.remove("hidden");
      try {
        const res = await fetch(`${API_BASE}/api/movies?query=${encodeURIComponent(val)}&limit=10`);
        const data = await res.json();
        renderAutocomplete(data.movies || []);
      } catch (err) {
        console.error("Lỗi tìm kiếm phim:", err);
      } finally {
        searchSpinner.classList.add("hidden");
      }
    }, 250);
  });

  function renderAutocomplete(movies) {
    if (!movies || movies.length === 0) {
      autocompleteList.innerHTML = `<div class="autocomplete-item"><span class="ac-title">Không tìm thấy phim phù hợp</span></div>`;
      autocompleteList.classList.remove("hidden");
      return;
    }

    autocompleteList.innerHTML = movies.map(m => `
      <div class="autocomplete-item" data-id="${m.movieId}" data-title="${escapeHtml(m.title)}">
        <span class="ac-title">${escapeHtml(m.title)}</span>
        <span class="ac-genres">${escapeHtml(m.genres)}</span>
      </div>
    `).join("");

    autocompleteList.classList.remove("hidden");

    // Click chọn phim từ autocomplete
    autocompleteList.querySelectorAll(".autocomplete-item").forEach(item => {
      item.addEventListener("click", () => {
        const id = parseInt(item.getAttribute("data-id"));
        const title = item.getAttribute("data-title");
        if (id) {
          selectedMovieId = id;
          searchInput.value = title;
          autocompleteList.classList.add("hidden");
          executeRecommendation(selectedMovieId);
        }
      });
    });
  }

  // Click ra ngoài thì đóng dropdown
  document.addEventListener("click", (e) => {
    if (!searchInput.contains(e.target) && !autocompleteList.contains(e.target)) {
      autocompleteList.classList.add("hidden");
    }
  });

  // ==================== QUICK SEED CHIPS ====================
  document.querySelectorAll(".seed-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const id = parseInt(chip.getAttribute("data-id"));
      const title = chip.getAttribute("data-title");
      selectedMovieId = id;
      searchInput.value = title;
      autocompleteList.classList.add("hidden");
      executeRecommendation(id);
    });
  });

  // ==================== RECOMMENDATION ACTION ====================
  btnRecommend.addEventListener("click", () => {
    if (selectedMovieId) {
      executeRecommendation(selectedMovieId);
    } else {
      // Tìm thử tên trong input
      const query = searchInput.value.trim();
      if (!query) {
        alert("Vui lòng nhập tên phim hoặc chọn một phim để bắt đầu!");
        return;
      }
      fetch(`${API_BASE}/api/movies?query=${encodeURIComponent(query)}&limit=1`)
        .then(res => res.json())
        .then(data => {
          if (data.movies && data.movies.length > 0) {
            selectedMovieId = data.movies[0].movieId;
            executeRecommendation(selectedMovieId);
          } else {
            alert("Không tìm thấy phim này trong cơ sở dữ liệu!");
          }
        });
    }
  });

  genreFilter.addEventListener("change", () => {
    if (selectedMovieId) executeRecommendation(selectedMovieId);
  });

  kSelect.addEventListener("change", () => {
    if (selectedMovieId) executeRecommendation(selectedMovieId);
  });

  async function executeRecommendation(movieId) {
    const k = parseInt(kSelect.value) || 10;
    const genre = genreFilter.value;
    
    // Hiển thị trạng thái đang tải
    recsGrid.innerHTML = `
      <div class="empty-state">
        <div class="spinner" style="position:static; margin:0 auto 1rem; width:36px; height:36px;"></div>
        <h4>Đang tính toán độ tương đồng Cosine...</h4>
        <p>Đang truy vấn ma trận thưa Item-User trên không gian 610 chiều.</p>
      </div>
    `;

    const startTime = performance.now();

    try {
      // 1. Lấy chi tiết seed movie
      const movieRes = await fetch(`${API_BASE}/api/movies/${movieId}`);
      if (movieRes.ok) {
        const movieData = await movieRes.json();
        seedTitle.textContent = movieData.title;
        seedGenres.textContent = movieData.genres;
        seedRatings.textContent = movieData.rating_count ? movieData.rating_count.toLocaleString() : "0";
        seedAvgRating.textContent = movieData.rating_mean ? movieData.rating_mean.toFixed(1) + " / 5.0" : "N/A";
        seedCard.classList.remove("hidden");
      }

      // 2. Lấy gợi ý
      let url = `${API_BASE}/api/recommendations?movie_id=${movieId}&k=${k}`;
      if (genre && genre !== "all") {
        url += `&genre=${encodeURIComponent(genre)}`;
      }

      const recRes = await fetch(url);
      const recData = await recRes.json();
      const elapsed = Math.round(performance.now() - startTime);

      queryMetaInfo.textContent = `Độ trễ API: ${elapsed}ms | K=${k}`;

      // Xử lý cảnh báo (Warning) nếu dữ liệu thưa
      if (recData.warning) {
        seedWarningMsg.textContent = recData.warning;
        seedWarningBox.classList.remove("hidden");
      } else {
        seedWarningBox.classList.add("hidden");
      }

      renderRecommendations(recData.recommendations || []);
    } catch (err) {
      console.error("Lỗi khi tải gợi ý:", err);
      recsGrid.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h4>Lỗi xử lý yêu cầu</h4>
          <p>${err.message || "Vui lòng đảm bảo server đang chạy và model đã được nạp."}</p>
        </div>
      `;
    }
  }

  function renderRecommendations(items) {
    if (!items || items.length === 0) {
      recsGrid.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🔍</div>
          <h4>Không tìm thấy phim tương đồng phù hợp</h4>
          <p>Phim này có thể có quá ít tương tác hoặc không khớp với bộ lọc thể loại đã chọn.</p>
        </div>
      `;
      recsCountTitle.textContent = "Không có kết quả";
      return;
    }

    recsCountTitle.textContent = `Tìm thấy ${items.length} phim tương đồng hàng đầu`;

    recsGrid.innerHTML = items.map((item, idx) => {
      const simPercent = Math.min(100, Math.max(0, Math.round(item.similarity_score * 100)));
      return `
        <div class="rec-card">
          <div>
            <div class="rec-top">
              <span class="rec-rank">#${idx + 1}</span>
              <div class="rec-score-box">
                <span class="rec-score-val">${item.similarity_score.toFixed(4)}</span>
                <span class="rec-score-lbl">Cosine Sim</span>
              </div>
            </div>
            <h4 class="rec-title">${escapeHtml(item.title)}</h4>
            <div class="rec-genres">${escapeHtml(item.genres)}</div>
            
            <div class="sim-bar-container" title="Độ tương đồng: ${simPercent}%">
              <div class="sim-bar-fill" style="width: ${simPercent}%"></div>
            </div>

            <div class="rec-reason">
              💡 ${escapeHtml(item.reason)}
            </div>
          </div>

          <div class="rec-bottom">
            <span>⭐ Rating: <strong>${item.rating_mean.toFixed(1)}</strong>/5.0</span>
            <span>🗳️ <strong>${item.rating_count.toLocaleString()}</strong> đánh giá</span>
          </div>
        </div>
      `;
    }).join("");
  }

  // ==================== DASHBOARD: 10 FAILURE CASES ====================
  async function loadFailureCases() {
    try {
      const res = await fetch(`${API_BASE}/api/failure-cases`);
      if (!res.ok) return;
      const data = await res.json();
      
      failureTableBody.innerHTML = data.map(item => `
        <tr>
          <td><code>${item.movieId}</code></td>
          <td><strong>${escapeHtml(item.title)}</strong></td>
          <td><span class="badge ${item.case_type.includes('Bom') ? 'badge-primary' : 'badge-warning'}">${escapeHtml(item.case_type)}</span></td>
          <td>${escapeHtml(item.top_1_sim)}</td>
          <td><code>${typeof item.top_1_score === 'number' ? item.top_1_score.toFixed(4) : item.top_1_score}</code></td>
          <td><small style="color:var(--text-dim);">${escapeHtml(item.identified_issues.join(" | "))}</small></td>
        </tr>
      `).join("");
    } catch (err) {
      console.warn("Chưa tải được bảng 10 ca lỗi:", err);
    }
  }

  if (btnReloadFailures) {
    btnReloadFailures.addEventListener("click", loadFailureCases);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.toString()
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Tự động load phim mẫu mặc định: Toy Story (movieId: 1)
  selectedMovieId = 1;
  executeRecommendation(1);
});
