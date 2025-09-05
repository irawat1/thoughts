# Thoughts - Data Science Portfolio

A data science portfolio website built with Jekyll and the So Simple theme, featuring Jupyter notebooks and projects.

## Features

- **Clean Design**: Built with the So Simple Jekyll theme
- **Jupyter Notebooks**: Showcase your data science projects and analyses
- **Responsive**: Works on all devices
- **GitHub Pages**: Free hosting on GitHub Pages
- **Search**: Built-in search functionality
- **Categories & Tags**: Organize your content
- **Social Links**: Connect your social profiles

## Getting Started

### Prerequisites

- Ruby (2.7 or higher)
- Bundler gem
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/irawat1/thoughts.git
   cd thoughts
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Serve the site locally:
   ```bash
   bundle exec jekyll serve
   ```

4. Open your browser to `http://localhost:4000`

## Customization

### Personal Information

Edit `_config.yml` to update:
- Site title and description
- Author information
- Social media links
- Contact information

### Adding Notebooks

1. Create a new file in `_notebooks/` with the format `YYYY-MM-DD-title.md`
2. Add front matter with metadata:
   ```yaml
   ---
   layout: notebook
   title: "Your Notebook Title"
   date: 2024-01-15
   tags: [python, data-analysis]
   categories: [Data Analysis]
   excerpt: "Brief description of your notebook"
   ---
   ```

### Adding Pages

Create new pages in `_pages/` directory with appropriate front matter.

### Styling

Customize the theme by:
1. Creating `assets/css/main.scss` to override default styles
2. Modifying `_sass/` files for more extensive changes

## Deployment

This site is configured for GitHub Pages deployment:

1. Push your changes to the `main` branch
2. GitHub Pages will automatically build and deploy your site
3. Your site will be available at `https://irawat1.github.io/thoughts`

## Content Structure

```
thoughts/
├── _config.yml          # Site configuration
├── _notebooks/          # Jupyter notebook posts
├── _posts/              # Regular blog posts
├── _pages/              # Static pages
├── assets/              # Images, CSS, JS
└── index.md             # Homepage
```

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Feel free to fork this repository and customize it for your own portfolio!

## Support

If you have any questions or need help customizing your portfolio, please open an issue on GitHub.