/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef elxImpactMetric_hxx
#define elxImpactMetric_hxx

#include "elxImpactMetric.h"
#include "itkTimeProbe.h"
#include <vector>
#include <sstream>
#include <filesystem>
#include <algorithm>


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */
template <typename TElastix>
void
ImpactMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  // Log all model configurations (fixed and moving) with full detail.
  // This helps verify that the model settings are correctly parsed and applied.
  std::ostringstream oss;
  oss << "Initialization of Impact metric took: " << static_cast<int64_t>(timer.GetMean() * 1000)
      << " ms with \nFixed : ";
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); ++i)
  {
    oss << "\n\tModel(" << i << ") : \n" << this->GetFixedModelsConfiguration()[i];
  }
  oss << "\nMoving : ";
  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); ++i)
  {
    oss << "\n\tModel(" << i << ") : \n" << this->GetMovingModelsConfiguration()[i];
  }

  oss << "\nSubsetFeatures: " << GetStringFromVector<unsigned int>(this->GetSubsetFeatures())
      << "\nPCA: " << GetStringFromVector<unsigned int>(this->GetPCA())
      << "\nLayersWeight: " << GetStringFromVector<float>(this->GetLayersWeight())
      << "\nDistance: " << GetStringFromVector<std::string>(this->GetDistance()) << "\nMode: " << this->GetMode()
      << "\nDevice: " << this->GetDevice();

  if (this->GetMode() == "Static")
  {
    oss << "\nFeaturesMapUpdateInterval: " << this->GetFeaturesMapUpdateInterval()
        << "\n WriteFeatureMaps: " << this->GetWriteFeatureMaps();
    if (this->GetWriteFeatureMaps())
    {
      oss << "\n FeatureMapsPath: " << this->GetFeatureMapsPath();
    }
  }

  log::info(oss.str());
} // end Initialize()


/**
 * ******************* GenerateModelsConfiguration ***********************
 */
template <typename TElastix>
std::vector<typename ImpactMetric<TElastix>::Superclass1::ModelConfiguration>
ImpactMetric<TElastix>::GenerateModelsConfiguration(unsigned int level,
                                                    std::string  prefix,
                                                    std::string  mode,
                                                    unsigned int imageDimension)
{
  std::vector<typename ImpactMetric<TElastix>::Superclass1::ModelConfiguration> modelsConfiguration;

  /** Get and set the model path. */
  std::string modelsPathStr;
  this->GetConfiguration()->ReadParameter(modelsPathStr, prefix + "ModelsPath", this->GetComponentLabel(), level, 0);
  std::vector<std::string> modelsPathVec = GetVectorFromString<std::string>(modelsPathStr, "Path");

  /** Get and set the model dimension. */
  std::string modelDimension;
  this->GetConfiguration()->ReadParameter(modelDimension, prefix + "Dimension", this->GetComponentLabel(), level, 0);
  std::vector<unsigned int> modelsDimensionVec =
    GetVectorFromString<unsigned int>(modelsPathVec.size(), modelDimension, 3);

  /** Get and set the number of channels in model entry. */
  std::string numberOfChannels;
  this->GetConfiguration()->ReadParameter(
    numberOfChannels, prefix + "NumberOfChannels", this->GetComponentLabel(), level, 0);
  std::vector<unsigned int> numberOfChannelsVec =
    GetVectorFromString<unsigned int>(modelsPathVec.size(), numberOfChannels, 1);

  /** Get and set the voxel size. */
  std::string               patchSizeStr;
  std::vector<unsigned int> dimensions;
  if (mode == "Static")
  {
    patchSizeStr = formatParameterStringByDimensionAndLevel<Configuration>(
      this->GetConfiguration(), prefix, "PatchSize", level, imageDimension);
    for (int i = 0; i < modelsPathVec.size(); ++i)
      dimensions.push_back(imageDimension);
    if (patchSizeStr.empty() && level > 0)
    {
      patchSizeStr = formatParameterStringByDimensionAndLevel<Configuration>(
        this->GetConfiguration(), prefix, "PatchSize", 0, imageDimension);
    }
  }
  else
  {
    patchSizeStr =
      formatParameterStringByDimensionAndLevel<Configuration>(this->GetConfiguration(), prefix, "PatchSize", level);
    dimensions = modelsDimensionVec;
    if (patchSizeStr.empty() && level > 0)
    {
      patchSizeStr =
        formatParameterStringByDimensionAndLevel<Configuration>(this->GetConfiguration(), prefix, "PatchSize", 0);
    }
  }
  std::vector<std::string> patchSizeVec = groupStrByDimensions(patchSizeStr, dimensions);

  std::string              voxelSizeStr;
  std::vector<std::string> voxelSizeVec;
  if (mode == "Static")
  {
    voxelSizeStr = formatParameterStringByDimensionAndLevel<Configuration>(
      this->GetConfiguration(), prefix, "VoxelSize", level, imageDimension);
    std::vector<unsigned int> imageDimensions;
    for (int i = 0; i < modelsPathVec.size(); ++i)
      imageDimensions.push_back(imageDimension);
    voxelSizeVec = groupStrByDimensions(voxelSizeStr, imageDimensions);
  }
  else
  {
    voxelSizeStr =
      formatParameterStringByDimensionAndLevel<Configuration>(this->GetConfiguration(), prefix, "VoxelSize", level);
    voxelSizeVec = groupStrByDimensions(voxelSizeStr, modelsDimensionVec);
  }


  /** Get and set the Strides. */
  std::string layersMaskStr;
  this->GetConfiguration()->ReadParameter(layersMaskStr, prefix + "LayersMask", this->GetComponentLabel(), level, 0);
  std::vector<std::string> layersMaskVec = GetVectorFromString<std::string>(modelsPathVec.size(), layersMaskStr, "1");

  // Build the ModelConfiguration object for each model.
  // Each configuration includes model path, input dimension, channel count,
  // patch size, voxel size, and layer mask.
  // In static mode, we flag the model to cache features at init.
  for (int i = 0; i < modelsPathVec.size(); ++i)
  {
    try
    {

      modelsConfiguration.emplace_back(
        modelsPathVec[i],
        modelsDimensionVec[i],
        numberOfChannelsVec[i],
        GetVectorFromString<unsigned int>(modelsDimensionVec[i], patchSizeVec[i], 5),
        GetVectorFromString<float>(mode == "Static" ? imageDimension : modelsDimensionVec[i], voxelSizeVec[i], 1.5),
        GetBooleanVectorFromString(layersMaskVec[i], false),
        mode == "Static");
    }
    catch (const c10::Error & e)
    {
      itkExceptionMacro("ERROR: the model are not loaded from this file: " << modelsPathVec[i]
                                                                           << ". Torch error message:" << e.what());
    }
  }
  return modelsConfiguration;
} // end GenerateModelsConfiguration

/**
 * ***************** BeforeEachResolution ***********************
 * Read user-specified configuration for model, loss type, etc.
 */
template <typename TElastix>
void
ImpactMetric<TElastix>::BeforeEachResolution()
{
  this->m_CurrentIteration = 0;

  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();
  this->SetCurrentLevel(level);

  unsigned int randomSeed = 0;
  this->GetConfiguration()->ReadParameter(randomSeed, "RandomSeed", 0, false);
  this->SetSeed(randomSeed);

  // Read the mode of operation for the metric: "Jacobian" or "Static".
  // - Static: features are precomputed and optionally saved.
  // - Jacobian: gradients are propagated through the models.
  std::string mode = "Jacobian";
  this->GetConfiguration()->ReadParameter(mode, "ImpactMode", this->GetComponentLabel(), level, 0);
  if (mode != "Jacobian" && mode != "Static")
  {
    itkExceptionMacro(
      "Invalid mode: '" << mode << "'. Supported modes are 'Jacobian' and 'Static'. Please check the configuration.");
  }
  this->SetMode(mode);

  // Generate configuration for fixed and moving images.
  // Priority: FixedModelsPath > MovingModelsPath > ModelsPath
  // This allows shared or asymmetric feature extractors.
  // Try to read shared model path (used if FixedModelsPath / MovingModelsPath are not provided)
  std::string modelsPath;
  bool        hasSharedModel =
    this->GetConfiguration()->ReadParameter(modelsPath, "ImpactModelsPath", this->GetComponentLabel(), level, 0);

  // Generate fixed model configuration (fallback to shared if FixedModelsPath is missing)
  bool hasFixed =
    this->GetConfiguration()->ReadParameter(modelsPath, "ImpactFixedModelsPath", this->GetComponentLabel(), level, 0);
  if (hasFixed)
  {
    this->SetFixedModelsConfiguration(
      this->GenerateModelsConfiguration(level, "ImpactFixed", mode, FixedImageDimension));
  }
  else if (hasSharedModel)
  {
    this->SetFixedModelsConfiguration(this->GenerateModelsConfiguration(level, "Impact", mode, FixedImageDimension));
  }
  else
  {
    itkExceptionMacro("Missing parameter: FixedModelsPath or shared ModelsPath must be provided.");
  }

  // Generate moving model configuration (fallback to shared if MovingModelsPath is missing)
  bool hasMoving =
    this->GetConfiguration()->ReadParameter(modelsPath, "ImpactMovingModelsPath", this->GetComponentLabel(), level, 0);
  if (hasMoving)
  {
    this->SetMovingModelsConfiguration(
      this->GenerateModelsConfiguration(level, "ImpactMoving", mode, MovingImageDimension));
  }
  else if (hasSharedModel)
  {
    this->SetMovingModelsConfiguration(this->GenerateModelsConfiguration(level, "Impact", mode, MovingImageDimension));
  }
  else
  {
    itkExceptionMacro("Missing parameter: MovingModelsPath or shared ModelsPath must be provided.");
  }

  // Sanity check: models must not exceed image dimensionality
  // Useful to catch errors with 3D models on 2D images, for example.
  if (this->GetMode() == "Jacobian" &&
      this->GetFixedModelsConfiguration().size() != this->GetMovingModelsConfiguration().size())
  {
    itkExceptionMacro("Error: In 'Jacobian' mode, the number of fixed and moving models must be the same. Got "
                      << this->GetFixedModelsConfiguration().size() << " fixed model(s) and "
                      << this->GetMovingModelsConfiguration().size() << " moving model(s).");
  }

  // Ensure model input dimensions are not higher than image dimensions (e.g., model dim 3 vs image dim 2)
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); ++i)
  {
    if (this->GetFixedModelsConfiguration()[i].GetDimension() > FixedImageDimension)
    {
      itkExceptionMacro("ERROR: The dimension of the fixed input model image exceeds the allowed image dimensions. "
                        "Expected a maximum of "
                        << FixedImageDimension << " dimension(s), but received "
                        << this->GetFixedModelsConfiguration()[i].GetDimension() << " dimension(s) for model index "
                        << i
                        << ". "
                           "Please verify the input model dimensions.");
    }
  }

  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); ++i)
  {
    if (this->GetMovingModelsConfiguration()[i].GetDimension() > FixedImageDimension)
    {
      itkExceptionMacro("ERROR: The dimension of the moving input model image exceeds the allowed image dimensions. "
                        "Expected a maximum of "
                        << MovingImageDimension << " dimension(s), but received "
                        << this->GetMovingModelsConfiguration()[i].GetDimension() << " dimension(s) for model index "
                        << i
                        << ". "
                           "Please verify the input model dimensions.");
    }
  }

  // Choose GPU device if available and requested, fallback to CPU otherwise.
  // Raise explicit errors if user-requested GPU index is invalid.
  int device = -1;
  this->GetConfiguration()->ReadParameter(device, "ImpactGPU", this->GetComponentLabel(), level, 0);

  // Select computation device (GPU or CPU) based on availability and config
  if (device >= 0)
  {
    if (torch::cuda::is_available())
    {
      int availableGPUs = torch::cuda::device_count();
      if (device < availableGPUs)
      {
        this->SetDevice(torch::Device(torch::kCUDA, device));
      }
      else
      {
        itkExceptionMacro("Requested GPU " << device << " is out of range. Only " << availableGPUs
                                           << " GPUs are available.");
      }
    }
    else
    {
      itkExceptionMacro("CUDA is not available. Please check your CUDA installation or run on a compatible device.");
    }
  }
  else
  {
    this->SetDevice(torch::Device(torch::kCPU));
  }

  // Handle feature map export setup.
  // If WriteFeatureMaps is set (either "true" or a path), create the directory.
  // Store the path to be reused during feature writing.
  if (mode == "Static")
  {
    int featuresMapUpdateInterval = -1;
    this->GetConfiguration()->ReadParameter(
      featuresMapUpdateInterval, "ImpactFeaturesMapUpdateInterval", this->GetComponentLabel(), level, 0);
    this->SetFeaturesMapUpdateInterval(featuresMapUpdateInterval);
  }

  //
  int fixedNumberOfLayers = 0;
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); ++i)
  {
    std::vector<bool> layersMask = this->GetFixedModelsConfiguration()[i].GetLayersMask();
    fixedNumberOfLayers += std::count(layersMask.begin(), layersMask.end(), true);
    this->GetFixedModelsConfiguration()[i].GetModel()->to(this->GetDevice());
  }

  int movingNumberOfLayers = 0;
  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); ++i)
  {
    std::vector<bool> layersMask = this->GetMovingModelsConfiguration()[i].GetLayersMask();
    movingNumberOfLayers += std::count(layersMask.begin(), layersMask.end(), true);
    this->GetMovingModelsConfiguration()[i].GetModel()->to(this->GetDevice());
  }

  if (fixedNumberOfLayers != movingNumberOfLayers)
  {
    itkExceptionMacro("Error: The number of layers in the fixed models ("
                      << fixedNumberOfLayers << ") does not match the number of layers in the moving model ("
                      << movingNumberOfLayers << "). Please ensure that the models are compatible.");
  }
  if (fixedNumberOfLayers == 0)
  {
    itkExceptionMacro("Error: At least one layer must be selected for comparison. "
                      "Please ensure that the configuration includes at least one layer to be compared.");
  }
  /** Get and set the SubsetFeatures. */
  std::string subsetFeaturesStr;
  this->GetConfiguration()->ReadParameter(
    subsetFeaturesStr, "ImpactSubsetFeatures", this->GetComponentLabel(), level, 0);
  this->SetSubsetFeatures(GetVectorFromString<unsigned int>(fixedNumberOfLayers, subsetFeaturesStr, 32));

  /** Get and set the SubsetFeatures. */
  std::string pcaStr;
  this->GetConfiguration()->ReadParameter(pcaStr, "ImpactPCA", this->GetComponentLabel(), level, 0);
  this->SetPCA(GetVectorFromString<unsigned int>(fixedNumberOfLayers, pcaStr, 0));

  /** Get and set the LayersWeight. */
  std::string layersWeightStr;
  this->GetConfiguration()->ReadParameter(layersWeightStr, "ImpactLayersWeight", this->GetComponentLabel(), level, 0);
  this->SetLayersWeight(GetVectorFromString<float>(fixedNumberOfLayers, layersWeightStr, 1.0));
  this->SetWriteFeatureMaps(false);

  if (mode == "Static")
  {
    std::string writeFeatureMapsStr = "false";
    this->GetConfiguration()->ReadParameter(
      writeFeatureMapsStr, "ImpactWriteFeatureMaps", this->GetComponentLabel(), level, 0);
    if (writeFeatureMapsStr != "false")
    {
      // If enabled, prepare output directory for feature map export (Static mode)
      if (!std::filesystem::exists(writeFeatureMapsStr))
      {
        try
        {
          std::filesystem::create_directories(writeFeatureMapsStr);
          std::filesystem::permissions(writeFeatureMapsStr,
                                       std::filesystem::perms::owner_all | std::filesystem::perms::group_all |
                                         std::filesystem::perms::others_all,
                                       std::filesystem::perm_options::replace);
          this->SetWriteFeatureMaps(true);
          this->SetFeatureMapsPath(writeFeatureMapsStr);
        }
        catch (std::filesystem::filesystem_error & e)
        {
          itkExceptionMacro("Error creating directory for feature maps: " << writeFeatureMapsStr << "\n"
                                                                          << "Exception: " << e.what());
        }
      }
      else
      {
        this->SetWriteFeatureMaps(true);
        this->SetFeatureMapsPath(writeFeatureMapsStr);
      }
    }
  }

  // Get and set the distances.
  std::string distanceStr;
  this->GetConfiguration()->ReadParameter(distanceStr, "ImpactDistance", this->GetComponentLabel(), level, 0);
  this->SetDistance(GetVectorFromString<std::string>(fixedNumberOfLayers, distanceStr, "L2"));
} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration ***********************
 */
template <typename TElastix>
void
ImpactMetric<TElastix>::AfterEachIteration()
{
  // In static mode, optionally update the moving feature maps during optimization.
  // This allows hybrid modes where features are refreshed every N iterations.
  this->m_CurrentIteration++;
  if (this->GetMode() == "Static" && this->GetFeaturesMapUpdateInterval() > 0 &&
      this->m_CurrentIteration % this->GetFeaturesMapUpdateInterval() == 0)
  {
    this->UpdateMovingFeaturesMaps();
  }
} // end AfterEachIteration()


} // end namespace elastix

#endif // end #ifndef elxImpactMetric_hxx
