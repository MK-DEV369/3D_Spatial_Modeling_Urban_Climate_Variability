/**
 * Physical constants and parameters for urban climate modeling
 * All values are SI units or explicitly noted
 */

export const PHYS = {
    // Building physics
    FLOOR_HEIGHT_M: 3,                    // Standard floor height in meters
    DEFAULT_BUILDING_HEIGHT_M: 12,        // Default: 4 floors (when no OSM data)
    BUILDING_LOAD_HEIGHT_M: 50000,        // Camera height below which buildings load (meters)

    // Urban Heat Island (UHI) parameters
    TREE_COOLING_C_PER_10PCT: 0.6,        // Temperature reduction per 10% canopy cover (°C)
    UHI_BUILDUP_C_PER_DENSITY: 0.4,       // UHI intensity per unit building density (°C)
    WATER_COOLING_C: 0.8,                 // Cooling effect of water bodies (°C)

    // Atmospheric parameters
    LAPSE_RATE_C_PER_100M: 0.65,          // Temperature decrease with elevation (°C/100m)
    REFERENCE_PRESSURE_MB: 1013.25,       // Sea level pressure (millibars)

    // Traffic flow parameters
    FREE_FLOW_SPEED_KMH: 60,              // Free flow speed on urban roads (km/h)
    CONGESTION_THRESHOLD_VPH: 1800,       // Vehicles per hour for congestion onset

    // Scenario multipliers (unitless)
    BASELINE_SCENARIO: {
        treeFactor: 1.0,
        densityFactor: 1.0,
        treeCoolingC: 0.0,
    },

    GREEN_SCENARIO: {
        treeFactor: 1.2,                    // 20% increase in vegetation
        densityFactor: 1.0,
        treeCoolingC: 0.6,                  // Additional cooling from trees
    },

    DENSE_SCENARIO: {
        treeFactor: 0.9,                    // 10% decrease in vegetation
        densityFactor: 1.3,                 // 30% increase in building density
        treeCoolingC: -0.3,                 // Reduced cooling (vegetation loss)
    },
}

export interface ScenarioParams {
    treeFactor: number          // Vegetation density multiplier
    densityFactor: number        // Building density multiplier
    treeCoolingC: number         // Temperature offset from vegetation (°C)
}

export function getScenarioParams(scenario: 'baseline' | 'green' | 'dense'): ScenarioParams {
    switch (scenario) {
        case 'green':
            return PHYS.GREEN_SCENARIO
        case 'dense':
            return PHYS.DENSE_SCENARIO
        case 'baseline':
        default:
            return PHYS.BASELINE_SCENARIO
    }
}

/**
 * Calculate local temperature with Urban Heat Island effect
 * @param baseTemp - Base temperature from weather API (°C)
 * @param buildingDensity - Building footprint ratio [0-1]
 * @param vegetationCover - Vegetation cover ratio [0-1]
 * @param elevation - Elevation above reference (meters)
 * @returns Adjusted local temperature (°C)
 */
export function calculateLocalTemperature(
    baseTemp: number,
    buildingDensity: number = 0.3,
    vegetationCover: number = 0.2,
    elevation: number = 0
): number {
    // UHI heating from buildings
    const uhiEffect = buildingDensity * PHYS.UHI_BUILDUP_C_PER_DENSITY

    // Cooling from vegetation
    const vegCooling = (vegetationCover / 0.1) * PHYS.TREE_COOLING_C_PER_10PCT

    // Elevation lapse rate
    const elevationEffect = -(elevation / 100) * PHYS.LAPSE_RATE_C_PER_100M

    return baseTemp + uhiEffect - vegCooling + elevationEffect
}

/**
 * Estimate building height from OSM properties
 * Priority: explicit height > levels > fallback
 */
export function estimateBuildingHeight(properties: any): number {
    // Try explicit height tag
    const heightTag = properties?.height?.getValue?.() ?? properties?.height
    if (heightTag && !isNaN(parseFloat(heightTag))) {
        return parseFloat(heightTag)
    }

    // Try building:levels tag
    const levelsTag = properties?.building_levels?.getValue?.() ??
        properties?.['building:levels']?.getValue?.() ??
        properties?.levels?.getValue?.() ??
        properties?.building_levels ??
        properties?.['building:levels'] ??
        properties?.levels

    if (levelsTag && !isNaN(parseInt(levelsTag))) {
        return parseInt(levelsTag) * PHYS.FLOOR_HEIGHT_M
    }

    // Fallback to default
    return PHYS.DEFAULT_BUILDING_HEIGHT_M
}