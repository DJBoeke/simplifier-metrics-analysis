import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pycountry
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
font_size = 14

# --- Load Data ---
df_public_packages = pd.read_csv(
    r"C:\Users\Gebruiker\OneDrive - Firely\SRP\Data analyse\Simplifier\publicPackages.csv",
    names=["Id", "PackageEntryId", "PackageName", "Version", "ReleaseDate", "ReleaseNotes", "Description",
           "Prerelease", "FhirVersion", "Unlisted", "Complete", "ShaSum", "Source", "PackageCanonical",
           "CanonicalState", "LastUpdate", "JurisdictionCode", "JurisdictionSystem", "InMainFeed", "ProjectId"],
    index_col=False
)

df_package_dependencies = pd.read_csv(
    r"C:\Users\Gebruiker\OneDrive - Firely\SRP\Data analyse\Simplifier\packageDependencies.csv",
    names=["Id", "SourcePackageID", "TargetPackageID", "FeedId"]
)
df_package_dependencies = df_package_dependencies[
    df_package_dependencies["SourcePackageID"].isin(df_public_packages["Id"]) &
    df_package_dependencies["TargetPackageID"].isin(df_public_packages["Id"])
]

df_download_packages = pd.read_csv(
    r"C:\Users\Gebruiker\OneDrive - Firely\SRP\Data analyse\Simplifier\allPackageDownloads.csv",
    names=["Id", "UserId", "Action", "SubjectId", "SubjectType", "Date_2"]
)
df_download_packages['Date_2'] = pd.to_datetime(df_download_packages['Date_2'], errors='coerce')
df_download_packages['Date'] = df_download_packages['Date_2'].dt.date
df_download_packages.drop(columns=['Date_2', 'UserId', 'Action'], inplace=True)

df_views_packages = pd.read_csv(
    r"C:\Users\Gebruiker\OneDrive - Firely\SRP\Data analyse\Simplifier\allPackageViews.csv",
    names=["Id", "UserId", "Action", "SubjectId", "SubjectType", "Date_2"]
)
df_views_packages['Date_2'] = pd.to_datetime(df_views_packages['Date_2'], errors='coerce')
df_views_packages['Date'] = df_views_packages['Date_2'].dt.date
df_views_packages.drop(columns=['Date_2', 'UserId', 'Action'], inplace=True)

print("Before script cleaning:", df_public_packages["PackageName"].dropna().nunique())

# --- Clean Data ---
replacements = {
    'urn:iso:std:iso:3166:GB-ENG': 'GB',
    'urn:iso:std:iso:3166:-2:GB-ENG': 'GB',
    '840': 'US',
    '276': 'DE'
}

df_public_packages['JurisdictionCode'] = df_public_packages['JurisdictionCode'].replace(replacements)
df_public_packages = df_public_packages.dropna(subset=["JurisdictionCode"])
df_public_packages = df_public_packages[~df_public_packages["JurisdictionCode"].isin(["001", "1"])]
df_public_packages = df_public_packages[
    (df_public_packages["Unlisted"] != 1) &
    (df_public_packages["Complete"] != 0) &
    (df_public_packages["Prerelease"] != 1) &
    (df_public_packages["Description"].notnull())
]

# --- Merge and Join ---
merged_downloads_df = pd.merge(
    df_download_packages,
    df_public_packages[["Id", "PackageEntryId", "PackageName", "JurisdictionCode", "Version", "LastUpdate"]],
    left_on="SubjectId", right_on="Id", how="inner"
).drop(columns=["Id_y"]).rename(columns={"Id_x": "Id"})

merged_views_df = pd.merge(
    df_views_packages,
    df_public_packages[["Id", "PackageEntryId", "PackageName", "JurisdictionCode", "Version", "LastUpdate"]],
    left_on="SubjectId", right_on="Id", how="inner"
).drop(columns=["Id_y"]).rename(columns={"Id_x": "Id"})

merged_package_dependencies = df_package_dependencies.merge(
    df_public_packages[['Id', 'PackageName']],
    left_on='SourcePackageID', right_on='Id', how='left'
).rename(columns={'PackageName': 'SourcePackageName'}).merge(
    df_public_packages[['Id', 'PackageName']],
    left_on='TargetPackageID', right_on='Id', how='left'
).rename(columns={'PackageName': 'TargetPackageName'}).drop(columns=['Id_x', 'Id_y'])

# Function to remove specific packages from the DataFrame
def remove_excluded_packages(df, exclude_list, column_name):
    exclude_list_lower = [pkg.lower() for pkg in exclude_list]
    return df[~df[column_name].str.lower().isin(exclude_list_lower)]


exclude_list = [
    # Terminology packages - Core terminology resources
    # Base packages - Generic templates without specific jurisdiction
    # Core FHIR packages provided by HL7 (international standard, not jurisdiction-specific)
    # Duplicate core packages from Simplifier (redundant with HL7 official packages)
    # Packages with high download/view counts but low-quality versions or pre-release status
    # Pre-release, demo, testing, or sandbox packages (not production-quality)
    # Invalid or template-only packages
    # Manually identified demo/test/sandbox packages (low quality or experimental)
    "ans.fhir.fr.tddui", "ans.fr.doctrine", "hl7.fhir.be.infsec", "junk.sample-preview", "kr.myfhirtest", "nat.testproject", "rapportendoscopiequebec.test",
    "sand.box", "test.module3.v2", "test.prova", "test.v202111591", "test20171286.neu", "test20211078.organization", "testproject130.com.banana","testprojekt.sl.r4",                    # Generic project identified as test (R4 version)
    "de.testprojektukf.rmy", "depar4.01", "ehelse.fhir.no.grunndata.test", "first.package", "patient-summary.setp_sandbox", "berkay.sandbox", "phis.ig.createtest", 
    "package.test.new", "laniado.test.fhir.r4", "ca.ec.demo.test", "careconnect.testpackage.stu3", "ch.cel.thetest2-core", "dummy.first", "mydummyproject.01", 
    "aws.dummy", "fhirtools.ig.template", "phis.ig.dev", "acme.profiling.tutorial.r4", "rl.fhir.r4.draft", "jp-core.draft1", "fhir.examplenov", 
    "devdays.r4.example.conformanceresources", "fhir-training.lab20", "jp-core.draft1", "ca.on.phsd.r4-alpha", "commonwell-consent-trial01.01", 
    "fmcna.caredata.fhir.ig.r4.copy", "ths-greifswald.ttp-fhir-gw", "tigacorehub.patient", "pathologyencountertissue.tryout", "prueba15.prueba", 
    "andersonsanto.tarefa6", "bla.abel.org", "project34.chdn.lu", "test.colo.qc", "test.public.project.pvt.package", "ntt_ir.r1.00.01"
    ]

# Sanity check before filtering
print("Before manual cleaning:", merged_downloads_df["PackageName"].dropna().nunique())

merged_downloads_df = remove_excluded_packages(merged_downloads_df, exclude_list, "PackageName")
merged_package_dependencies = remove_excluded_packages(merged_package_dependencies, exclude_list, "TargetPackageName")
merged_views_df = remove_excluded_packages(merged_views_df, exclude_list, "PackageName")

# Sanity check after filtering
print("After manual cleaning (merged_downloads_df):", merged_downloads_df["PackageName"].dropna().nunique())
print("After manual cleaning (merged_package_dependencies):", merged_package_dependencies["TargetPackageName"].dropna().nunique())
print("After manual cleaning (merged_views_df):", merged_views_df["PackageName"].dropna().nunique())

def list_all_packages(df_packages):
    df_packages["PackageName"] = df_packages["PackageName"].str.lower()
    return df_packages["PackageName"].unique()

# Get all packages
all_packages = list_all_packages(merged_downloads_df)

# Sorting the list of unique package names alphabetically
sorted_packages = sorted(all_packages)

# # Print sorted packages
# print("Sorted packages:")
# for pkg in sorted_packages:
#     print(pkg)

# --- Adoption Metrics ---

def plot_top5_bar_with_percentages(
    df_top5,
    value_col,
    metric_name,
    color,
    title,
    ylabel,
    fontsize=14,
    height_threshold=10
):
    """
    Plot a Top 5 bar chart with dynamic percentage labels and clean legend.
    """
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    bars = ax.bar(df_top5["JurisdictionCode"], df_top5[value_col], color=color, alpha=0.8, label=metric_name)

    # Add dynamic percentage labels
    for bar, (_, row) in zip(bars, df_top5.iterrows()):
        height = bar.get_height()
        if height > height_threshold:
            ax.text(bar.get_x() + bar.get_width()/2, height/2, f"{row['Percentage']:.1f}%",
                    ha='center', va='center', color='white', fontsize=fontsize-2, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height + 2, f"{row['Percentage']:.1f}%",
                    ha='center', va='bottom', color='black', fontsize=fontsize-2)

    ax.set_title(title, fontsize=fontsize+2, fontweight='bold')
    ax.set_xlabel("Jurisdiction Code", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='x', rotation=45, labelsize=fontsize-2)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a nice clean legend
    ax.legend(
        title="\n(%) = Global Share",
        fontsize=fontsize-2,
        title_fontsize=fontsize-2,
        loc='upper right',
        framealpha=0.7
    )

    plt.show()

# --- Count unique packages per jurisdiction ---
unique_packages_per_country = (
    merged_downloads_df.groupby("JurisdictionCode")["PackageName"]
    .nunique()
    .sort_values(ascending=False)
    .head(5)
)

total_unique_packages = merged_downloads_df["PackageName"].nunique()
top_unique_packages_df = unique_packages_per_country.reset_index(name="UniquePackages")
top_unique_packages_df["Percentage"] = (top_unique_packages_df["UniquePackages"] / total_unique_packages) * 100

# --- Plot Unique Packages ---
plot_top5_bar_with_percentages(
    df_top5=top_unique_packages_df,
    value_col="UniquePackages",
    metric_name="Unique Packages",
    color="#fcf403",
    title="Top 5 Jurisdictions by Unique Packages",
    ylabel="Number of Unique Packages"
)

# --- Count unique projects per jurisdiction ---
projects_per_country = (
    df_public_packages.groupby("JurisdictionCode")["ProjectId"]
    .nunique()
    .reset_index(name="TotalProjects")
    .sort_values(by="TotalProjects", ascending=False)
)

top_projects = projects_per_country.head(5).copy()
total_unique_projects = projects_per_country["TotalProjects"].sum()
top_projects["Percentage"] = (top_projects["TotalProjects"] / total_unique_projects) * 100

# --- Plot Unique Projects ---
plot_top5_bar_with_percentages(
    df_top5=top_projects,
    value_col="TotalProjects",
    metric_name="Unique Projects",
    color="#03fcdf",
    title="Top 5 Jurisdictions by Unique Projects",
    ylabel="Number of Unique Projects"
)

# --- Aggregate total downloads and views per jurisdiction ---
jurisdiction_activity = (
    merged_downloads_df.groupby("JurisdictionCode").size().reset_index(name="TotalDownloads")
    .merge(
        merged_views_df.groupby("JurisdictionCode").size().reset_index(name="TotalViews"),
        on="JurisdictionCode",
        how="outer"
    )
    .fillna(0)
)

jurisdiction_activity = jurisdiction_activity.merge(
    unique_packages_per_country.reset_index(name="UniquePackages"),
    on="JurisdictionCode",
    how="left"
).fillna(0)

# --- Top 5 Downloads ---
top_downloads = jurisdiction_activity.sort_values(by="TotalDownloads", ascending=False).head(5)
total_downloads = jurisdiction_activity["TotalDownloads"].sum()
top_downloads["Percentage"] = (top_downloads["TotalDownloads"] / total_downloads) * 100

# --- Plot Downloads ---
plot_top5_bar_with_percentages(
    df_top5=top_downloads,
    value_col="TotalDownloads",
    metric_name="Downloads",
    color="#1f77b4",
    title="Top 5 Jurisdictions by Downloads",
    ylabel="Total Downloads"
)

# --- Top 5 Views ---
top_views = jurisdiction_activity.sort_values(by="TotalViews", ascending=False).head(5)
total_views = jurisdiction_activity["TotalViews"].sum()
top_views["Percentage"] = (top_views["TotalViews"] / total_views) * 100

# --- Plot Views ---
plot_top5_bar_with_percentages(
    df_top5=top_views,
    value_col="TotalViews",
    metric_name="Views",
    color="#ff7f0e",
    title="Top 5 Jurisdictions by Views",
    ylabel="Total Views"
)

# --- Top 5 Dependencies ---
package_dependencies_df = merged_package_dependencies.rename(columns={"TargetPackageName": "PackageName"})
package_dependencies_df["JurisdictionCode"] = package_dependencies_df["PackageName"].map(
    merged_downloads_df.set_index("PackageName")["JurisdictionCode"].to_dict()
)
package_dependencies_df = package_dependencies_df.dropna(subset=["JurisdictionCode"])

dependencies_per_country = (
    package_dependencies_df.groupby("JurisdictionCode")
    .size()
    .reset_index(name="TotalDependencies")
    .sort_values(by="TotalDependencies", ascending=False)
)

top_dependencies = dependencies_per_country.head(5).copy()
total_dependencies = dependencies_per_country["TotalDependencies"].sum()
top_dependencies["Percentage"] = (top_dependencies["TotalDependencies"] / total_dependencies) * 100

# --- Plot Package Dependencies ---
plot_top5_bar_with_percentages(
    df_top5=top_dependencies,
    value_col="TotalDependencies",
    metric_name="Package Dependencies",
    color="#d62728",
    title="Top 5 Jurisdictions by Package Dependencies",
    ylabel="Total Dependencies"
)

# --- Adoption World Heatmap ---

#Downloads and Views
downloads = merged_downloads_df.groupby("JurisdictionCode").size().reset_index(name="TotalDownloads")
views = merged_views_df.groupby("JurisdictionCode").size().reset_index(name="TotalViews")
dependencies = package_dependencies_df.groupby("JurisdictionCode").size().reset_index(name="TotalDependencies")

# Aggregate all metrics by JurisdictionCode
fhir_adoption_df = df_public_packages.groupby("JurisdictionCode").agg(
    TotalProjects=("ProjectId", "nunique"),
    TotalPackages=("PackageName", "nunique")
).reset_index()

# Merge all into one
fhir_adoption_df = fhir_adoption_df.merge(downloads, on="JurisdictionCode", how="left")
fhir_adoption_df = fhir_adoption_df.merge(views, on="JurisdictionCode", how="left")
fhir_adoption_df = fhir_adoption_df.merge(dependencies, on="JurisdictionCode", how="left")

# Replace NaNs with 0s
fhir_adoption_df.fillna(0, inplace=True)

def alpha2_to_alpha3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None

fhir_adoption_df["JurisdictionCode"] = fhir_adoption_df["JurisdictionCode"].apply(alpha2_to_alpha3)
fhir_adoption_df.dropna(subset=["JurisdictionCode"], inplace=True)

# Create a new binary column: is the country active? (1 = active, 0 = not active)
fhir_adoption_df["IsActive"] = fhir_adoption_df["TotalPackages"].apply(lambda x: 1 if x > 0 else 0)

# Total number of active jurisdictions
total_active = fhir_adoption_df["IsActive"].sum()
print(f"Total active jurisdictions: {total_active}")

# Define a custom color scale for binary values
binary_colorscale = [
    [0, "white"],   # 0 = inactive -> white
    [1, "#922b21"]  # 1 = active -> your red color
]

# Plot the binary active map
fig = px.choropleth(
    fhir_adoption_df,
    locations="JurisdictionCode",
    locationmode="ISO-3",
    color="IsActive",
    hover_name="JurisdictionCode",
    color_continuous_scale=binary_colorscale,
    range_color=(0,1),
    title=f"FHIR Active Jurisdictions ({total_active} active)",
    projection="natural earth",
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_showscale=False  # Hide colorbar since it's binary
)

fig.show()

# --- Adoption Figure Heatmap ---

heatmap_data = fhir_adoption_df.set_index('JurisdictionCode')[
    ["TotalProjects", "TotalPackages", "TotalDownloads", "TotalViews", "TotalDependencies"]
]

# 2. Normalize each column individually
normalized_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

# 3. Calculate the mean of the normalized values across metrics
normalized_data["MeanMetricValue"] = normalized_data.mean(axis=1)

# 4. Select the Top 10 jurisdictions based on the normalized mean
top10_jurisdictions = normalized_data.sort_values("MeanMetricValue", ascending=False).head(10)

# 5. Drop the temporary MeanMetricValue column for heatmap plotting
top10_jurisdictions = top10_jurisdictions.drop(columns="MeanMetricValue")


# Plot the heatmap
plt.figure(figsize=(12, 8))  # You can adjust the figure size
sns.heatmap(
    top10_jurisdictions ,
    cmap="viridis",          
    annot=True,              # Show numbers inside the heatmap
    fmt=".2f",               # Format numbers with 2 decimals because it's normalized
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Normalized Adoption Metric'}
)
plt.title("FHIR Adoption Metrics", fontsize=16)
plt.xlabel("Adoption Metrics", fontsize=14)
plt.ylabel("Jurisdiction", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Adoption Trend Over Time ---
def plot_single_metric_trend(
    downloads_df,
    views_df,
    metric="downloads",   # 'downloads' or 'views'
    package_name=None,
    jurisdiction=None,
    event_dates=None,
    event_labels=None,
    window=7,
    font_size=12
):
    """
    Plot a single trend (downloads or views) over time for a specific package and/or jurisdiction.
    """
    df_downloads = downloads_df.copy()
    df_views = views_df.copy()

    # Apply filters
    if package_name:
        df_downloads = df_downloads[df_downloads["PackageName"] == package_name]
        df_views = df_views[df_views["PackageName"] == package_name]

    if jurisdiction:
        df_downloads = df_downloads[df_downloads["JurisdictionCode"] == jurisdiction]
        df_views = df_views[df_views["JurisdictionCode"] == jurisdiction]

    if df_downloads.empty and df_views.empty:
        print("No data for the specified filters.")
        return

    # Aggregate per day
    downloads_by_date = df_downloads.groupby("Date").size().reset_index(name="TotalDownloads")
    views_by_date = df_views.groupby("Date").size().reset_index(name="TotalViews")

    # Merge and fill missing dates
    trend_data = pd.merge(downloads_by_date, views_by_date, on="Date", how="outer").fillna(0)
    trend_data["Date"] = pd.to_datetime(trend_data["Date"])
    trend_data = trend_data.sort_values("Date")

    # Rolling averages
    trend_data["DownloadsRollingAvg"] = trend_data["TotalDownloads"].rolling(window=window, min_periods=1).mean()
    trend_data["ViewsRollingAvg"] = trend_data["TotalViews"].rolling(window=window, min_periods=1).mean()

    # Choose what to plot
    if metric == "downloads":
        y = trend_data["DownloadsRollingAvg"]
        label = "Avg. Downloads"
        color = "blue"
    elif metric == "views":
        y = trend_data["ViewsRollingAvg"]
        label = "Avg. Views"
        color = "green"
    else:
        raise ValueError("Invalid metric. Choose 'downloads' or 'views'.")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trend_data["Date"], y, label=label, color=color, linewidth=1.8)

    if event_dates and event_labels:
        for date, label_text in zip(event_dates, event_labels):
            ax.axvline(pd.to_datetime(date), color='red', linestyle='--', alpha=0.6)
            ax.text(pd.to_datetime(date), ax.get_ylim()[1], label_text, rotation=90, fontsize=10, verticalalignment='top')

    title_parts = []
    if package_name:
        title_parts.append(f"Package: {package_name}")
    if jurisdiction:
        title_parts.append(f"Jurisdiction: {jurisdiction}")
    title = " | ".join(title_parts) if title_parts else "Global"

    ax.set_title(f"FHIR {metric.capitalize()} Trend Over Time ({title})", fontsize=font_size+2, fontweight='bold')
    ax.set_xlabel("Date", fontsize=font_size)
    ax.set_ylabel(f"{metric.capitalize()}", fontsize=font_size)
    ax.legend(fontsize=font_size-2)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot only Downloads
plot_single_metric_trend(
    merged_downloads_df,
    merged_views_df,
    metric="downloads",
    jurisdiction="DE",
    event_dates=["2024-06-12", "2024-11-20"],
    event_labels=["Example 1", "Example 2"],
)

# Plot only Views
plot_single_metric_trend(
    merged_downloads_df,
    merged_views_df,
    metric="views",
    jurisdiction="DE",
    event_dates=["2024-06-12", "2024-11-20"],
    event_labels=["Example 1", "Example 2"],
)

# --- Adoption Comparison Trend Over Time ---

def compare_single_metric_trends(
    downloads_df,
    views_df,
    jurisdictions,
    metric="downloads",  # 'downloads' or 'views'
    package_name=None,
    event_dates=None,
    event_labels=None,
    window=7,
    font_size=12
):
    """
    Compare adoption trends (downloads or views) across jurisdictions for a given package (optional).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for code in jurisdictions:
        df_d = downloads_df.copy()
        df_v = views_df.copy()

        if package_name:
            df_d = df_d[df_d["PackageName"] == package_name]
            df_v = df_v[df_v["PackageName"] == package_name]

        df_d = df_d[df_d["JurisdictionCode"] == code]
        df_v = df_v[df_v["JurisdictionCode"] == code]

        if df_d.empty and df_v.empty:
            print(f"No data for {code}")
            continue

        # Aggregate daily counts
        d_by_date = df_d.groupby("Date").size().reset_index(name="Downloads")
        v_by_date = df_v.groupby("Date").size().reset_index(name="Views")
        merged = pd.merge(d_by_date, v_by_date, on="Date", how="outer").fillna(0)
        merged["Date"] = pd.to_datetime(merged["Date"])
        merged = merged.sort_values("Date")

        # Rolling averages
        merged["DownloadsAvg"] = merged["Downloads"].rolling(window=window, min_periods=1).mean()
        merged["ViewsAvg"] = merged["Views"].rolling(window=window, min_periods=1).mean()

        # Plot based on metric
        if metric == "downloads":
            ax.plot(merged["Date"], merged["DownloadsAvg"], label=f"{code} - Downloads", linewidth=1.8)
        elif metric == "views":
            ax.plot(merged["Date"], merged["ViewsAvg"], linestyle='--', label=f"{code} - Views", linewidth=1.8)
        else:
            raise ValueError("Invalid metric. Choose 'downloads' or 'views'.")

    # Event markers
    if event_dates and event_labels:
        for date, label_text in zip(event_dates, event_labels):
            ax.axvline(pd.to_datetime(date), color='red', linestyle='--', alpha=0.6)
            ax.text(pd.to_datetime(date), ax.get_ylim()[1], label_text, rotation=90, fontsize=10, verticalalignment='top')

    title = f"Comparative {metric.capitalize()} Trends"
    if package_name:
        title += f" | Package: {package_name}"

    ax.set_title(title, fontsize=font_size+2, fontweight='bold')
    ax.set_xlabel("Date", fontsize=font_size)
    ax.set_ylabel(f"{metric.capitalize()} (Smoothed)", fontsize=font_size)
    ax.legend(fontsize=font_size - 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Compare Downloads
compare_single_metric_trends(
    merged_downloads_df,
    merged_views_df,
    jurisdictions=["DE", "US", "CA", "GB", "NL"],
    metric="downloads",
    package_name=None,
    event_dates=["2024-06-12"],
    event_labels=["FHIR DevDays"]
)

# Compare Views
compare_single_metric_trends(
    merged_downloads_df,
    merged_views_df,
    jurisdictions=["DE", "US", "CA", "GB", "NL"],
    metric="views",
    package_name=None,
    event_dates=["2024-06-12"],
    event_labels=["FHIR DevDays"]
)

# --- Aggregate metrics for global maximums---

# Aggregate download counts per package
package_downloads = merged_downloads_df.groupby("PackageName").size().reset_index(name="TotalDownloads")

# Aggregate view counts per package
package_views = merged_views_df.groupby("PackageName").size().reset_index(name="TotalViews")

# Aggregate dependency count per package
package_dependencies_count = merged_package_dependencies.groupby("TargetPackageName").size().reset_index(name="DependencyCount")

# Global maximums
global_max_downloads = package_downloads["TotalDownloads"].max()
global_max_views = package_views["TotalViews"].max()
global_max_dependencies = package_dependencies_count["DependencyCount"].max()

# --- Adoption Metric per Country ---

def plot_adoption_metric(df, jurisdiction, metric, color, global_max):
    filtered_df = df[df['JurisdictionCode'] == jurisdiction].copy()

    if filtered_df.empty:
        print(f"No data for jurisdiction '{jurisdiction}'")
        return

    metric_count = filtered_df.groupby("PackageName").size().reset_index(name=metric)

    metric_count[f'Normalized{metric}'] = (
        metric_count[metric] / global_max if global_max > 0 else 0
    )

    metric_count = metric_count.sort_values(by=metric, ascending=False).head(7)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(metric_count['PackageName'], metric_count[f'Normalized{metric}'], color=color, alpha=0.8)

    ax.set_title(f"{metric} per Package in {jurisdiction} (Normalized by Global Max)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Package Name", fontsize=14)
    ax.set_ylabel(f"Normalized {metric}", fontsize=14)
    ax.set_xticks(range(len(metric_count['PackageName'])))
    ax.set_xticklabels(metric_count['PackageName'], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Germany
plot_adoption_metric(merged_downloads_df, 'DE', 'TotalDownloads', '#1f77b4', global_max_downloads)
plot_adoption_metric(merged_views_df, 'DE', 'TotalViews', '#2ca02c', global_max_views)
plot_adoption_metric(package_dependencies_df, 'DE', 'DependencyCount', '#d62728', global_max_dependencies)

# USA
plot_adoption_metric(merged_downloads_df, 'US', 'TotalDownloads', '#1f77b4', global_max_downloads)
plot_adoption_metric(merged_views_df, 'US', 'TotalViews', '#2ca02c', global_max_views)
plot_adoption_metric(package_dependencies_df, 'US', 'DependencyCount', '#d62728', global_max_dependencies)

# Canada
plot_adoption_metric(merged_downloads_df, 'CA', 'TotalDownloads', '#1f77b4', global_max_downloads)
plot_adoption_metric(merged_views_df, 'CA', 'TotalViews', '#2ca02c', global_max_views)
plot_adoption_metric(package_dependencies_df, 'CA', 'DependencyCount', '#d62728', global_max_dependencies)

# Great Britain
plot_adoption_metric(merged_downloads_df, 'GB', 'TotalDownloads', '#1f77b4', global_max_downloads)
plot_adoption_metric(merged_views_df, 'GB', 'TotalViews', '#2ca02c', global_max_views)
plot_adoption_metric(package_dependencies_df, 'GB', 'DependencyCount', '#d62728', global_max_dependencies)

# Netherlands
plot_adoption_metric(merged_downloads_df, 'NL', 'TotalDownloads', '#1f77b4', global_max_downloads)
plot_adoption_metric(merged_views_df, 'NL', 'TotalViews', '#2ca02c', global_max_views)
plot_adoption_metric(package_dependencies_df, 'NL', 'DependencyCount', '#d62728', global_max_dependencies)

# --- Package Metric Comparison ---

# Merge metrics into one table
package_metrics = package_downloads.merge(package_views, on="PackageName", how="outer")
package_metrics = package_metrics.merge(package_dependencies_count.rename(columns={"TargetPackageName": "PackageName"}), on="PackageName", how="outer")
package_metrics = package_metrics.fillna(0)

package_metrics["NormDownloads"] = package_metrics["TotalDownloads"] / global_max_downloads
package_metrics["NormViews"] = package_metrics["TotalViews"] / global_max_views
package_metrics["NormDependencies"] = package_metrics["DependencyCount"] / global_max_dependencies

# Function to create a bar plot for a given metric
def plot_top_metric(df, metric, title, color):
    top_df = df.sort_values(by=metric, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top_df["PackageName"], top_df[metric], color=color, alpha=0.8)

    ax.set_title(f"Top 15 Packages by {title}", fontsize=font_size + 4, fontweight='bold')
    ax.set_xlabel("Package Name", fontsize=font_size)
    ax.set_ylabel(title, fontsize=font_size)
    ax.set_xticks(range(len(top_df["PackageName"])))
    ax.set_xticklabels(top_df["PackageName"], rotation=45, ha='right', fontsize=font_size - 1)
    ax.tick_params(axis='y', labelsize=font_size - 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#Now generate the four plots
plot_top_metric(package_metrics, "NormDownloads", "Normalized Downloads", "#1f77b4")
plot_top_metric(package_metrics, "NormViews", "Normalized Views", "#2ca02c")
plot_top_metric(package_metrics, "NormDependencies", "Normalized Dependencies", "#d62728")