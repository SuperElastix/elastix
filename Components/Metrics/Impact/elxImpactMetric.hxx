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
  oss << "Initialization of Impact metric took: " << static_cast<long>(timer.GetMean() * 1000) << " ms with \nFixed : ";
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); i++)
  {
    oss << "\n\tModel(" << i << ") : \n"
        << "\t\tPath : " << this->GetFixedModelsConfiguration()[i].m_modelPath
        << "\n\t\tDimension : " << this->GetFixedModelsConfiguration()[i].m_dimension
        << "\n\t\tNumberOfChannels : " << this->GetFixedModelsConfiguration()[i].m_numberOfChannels;
    if (this->GetMode() != "Static")
    {
      oss << "\n\t\tPatchSize : "
          << this->GetStringFromVector<long>(this->GetFixedModelsConfiguration()[i].m_patchSize);
    }
    oss << "\n\t\tVoxelSize : " << this->GetStringFromVector<float>(this->GetFixedModelsConfiguration()[i].m_voxelSize)
        << "\n\t\tLayersMask : "
        << this->GetStringFromVector<bool>(this->GetFixedModelsConfiguration()[i].m_layersMask);
  }
  oss << "\nMoving : ";
  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); i++)
  {
    oss << "\n\tModel(" << i << ") : "
        << "\n\t\tPath : " << this->GetMovingModelsConfiguration()[i].m_modelPath
        << "\n\t\tDimension : " << this->GetMovingModelsConfiguration()[i].m_dimension
        << "\n\t\tNumberOfChannels : " << this->GetMovingModelsConfiguration()[i].m_numberOfChannels
        << "\n\t\tPatchSize : " << this->GetStringFromVector<long>(this->GetMovingModelsConfiguration()[i].m_patchSize)
        << "\n\t\tVoxelSize : " << this->GetStringFromVector<float>(this->GetMovingModelsConfiguration()[i].m_voxelSize)
        << "\n\t\tLayersMask : "
        << this->GetStringFromVector<bool>(this->GetMovingModelsConfiguration()[i].m_layersMask);
  }

  oss << "\nSubsetFeatures: " << this->GetStringFromVector<unsigned int>(this->GetSubsetFeatures())
      << "\nPCA: " << this->GetStringFromVector<unsigned int>(this->GetPCA())
      << "\nLayersWeight: " << this->GetStringFromVector<float>(this->GetLayersWeight())
      << "\nDistance: " << this->GetStringFromVector<std::string>(this->GetDistance()) << "\nMode: " << this->GetMode()
      << "\nGPU: " << this->GetGPU();

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
 * ******************* GetVectorFromString ***********************
 */
template <typename TElastix>
template <typename T>
std::vector<T>
ImpactMetric<TElastix>::GetVectorFromString(int size, std::string valueStr, T defaultValue)
{
  std::stringstream ss(valueStr);
  std::vector<T>    values;
  T                 value;

  while (ss >> value && values.size() < size)
  {
    values.push_back(value);
  }
  if (values.empty())
  {
    values.push_back(defaultValue);
  }
  while (values.size() < size)
  {
    values.push_back(values[0]);
  }
  return values;
} // end GetVectorFromString

/**
 * ******************* GetVectorFromString ***********************
 */
template <typename TElastix>
template <typename T>
std::vector<T>
ImpactMetric<TElastix>::GetVectorFromString(int size, std::string valueStr, T defaultValue, char delimiter)
{
  std::vector<T> values;

  if (delimiter == '\0')
  {
    for (char c : valueStr)
    {
      if (values.size() >= size)
        break;
      std::stringstream tokenStream(std::string(1, c));
      T                 value;
      if (tokenStream >> value)
      {
        values.push_back(value);
      }
    }
  }
  else
  {
    std::stringstream ss(valueStr);
    std::string       token;

    while (std::getline(ss, token, delimiter) && values.size() < size)
    {
      std::stringstream tokenStream(token);
      T                 value;
      if (tokenStream >> value)
      {
        values.push_back(value);
      }
    }
  }
  if (values.empty())
  {
    values.push_back(defaultValue);
  }
  while (values.size() < size)
  {
    values.push_back(values[0]);
  }

  return values;
} // end GetVectorFromString

/**
 * ******************* GetVectorFromString ***********************
 */
template <typename TElastix>
template <typename T>
std::vector<T>
ImpactMetric<TElastix>::GetVectorFromString(std::string valueStr, T defaultValue, char delimiter)
{
  std::vector<T> values;

  if (delimiter == '\0')
  {
    for (char c : valueStr)
    {
      std::stringstream tokenStream(std::string(1, c));
      std::string       token;
      if (tokenStream >> token)
      {
        if constexpr (std::is_unsigned<T>::value)
        {
          int temp = std::stoi(token);
          if (temp < 0)
          {
            temp = 0;
          }
          values.push_back(static_cast<T>(temp));
        }
        else
        {
          values.push_back(static_cast<T>(token));
        }
      }
    }
  }
  else
  {
    std::stringstream ss(valueStr);
    std::string       token;

    while (std::getline(ss, token, delimiter))
    {
      std::stringstream tokenStream(token);
      std::string       token1;
      if (tokenStream >> token1)
      {
        if constexpr (std::is_unsigned<T>::value)
        {
          int temp = std::stoi(token1);
          if (temp < 0)
          {
            temp = 0;
          }
          values.push_back(static_cast<T>(temp));
        }
        else
        {
          values.push_back(static_cast<T>(token1));
        }
      }
    }
  }
  if (values.empty())
  {
    values.push_back(defaultValue);
  }
  return values;
} // end GetVectorFromString

/**
 * ******************* GetVectorFromString ***********************
 */
template <typename TElastix>
template <typename T>
std::vector<T>
ImpactMetric<TElastix>::GetVectorFromString(std::string valueStr, T defaultValue)
{
  std::stringstream ss(valueStr);
  std::vector<T>    values;
  std::string       token;

  while (ss >> token)
  {
    if constexpr (std::is_unsigned<T>::value)
    {
      int temp = std::stoi(token);
      if (temp < 0)
      {
        temp = 0;
      }
      values.push_back(static_cast<T>(temp));
    }
    else
    {
      values.push_back(static_cast<T>(token));
    }
  }
  if (values.empty())
  {
    values.push_back(defaultValue);
  }
  return values;
} // end GetVectorFromString

/**
 * ******************* GetStringFromVector ***********************
 */
template <typename TElastix>
template <typename T>
std::string
ImpactMetric<TElastix>::GetStringFromVector(const std::vector<T> & vec)
{
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < vec.size(); ++i)
  {
    ss << vec[i];
    if (i != vec.size() - 1)
    {
      ss << " ";
    }
  }
  ss << ")";
  return ss.str();
} // end GetStringFromVector

/**
 * ******************* GenerateModelsConfiguration ***********************
 */
template <typename TElastix>
std::vector<typename ImpactMetric<TElastix>::Superclass1::ModelConfiguration>
ImpactMetric<TElastix>::GenerateModelsConfiguration(unsigned int level,
                                                    std::string  type,
                                                    std::string  mode,
                                                    unsigned int imageDimension)
{
  std::vector<typename ImpactMetric<TElastix>::Superclass1::ModelConfiguration> modelsConfiguration;

  /** Get and set the model path. */
  std::string modelsPathStr;
  this->GetConfiguration()->ReadParameter(modelsPathStr, type + "ModelsPath", this->GetComponentLabel(), level, 0);
  std::vector<std::string> modelsPathVec = this->GetVectorFromString<std::string>(modelsPathStr, "Path");
  if (modelsPathVec.empty())
  {
    itkExceptionMacro("Error: The parameter " + type + "ModelsPath is empty. Please check the configuration file.");
  }

  /** Get and set the model dimension. */
  std::string modelDimension;
  this->GetConfiguration()->ReadParameter(modelDimension, type + "Dimension", this->GetComponentLabel(), level, 0);
  std::vector<unsigned int> modelsDimensionVec =
    this->GetVectorFromString<unsigned int>(modelsPathVec.size(), modelDimension, 3);

  /** Get and set the number of channels in model entry. */
  std::string numberOfChannels;
  this->GetConfiguration()->ReadParameter(
    numberOfChannels, type + "NumberOfChannels", this->GetComponentLabel(), level, 0);
  std::vector<unsigned int> numberOfChannelsVec =
    this->GetVectorFromString<unsigned int>(modelsPathVec.size(), numberOfChannels, 1);
  std::vector<std::string> patchSizeVec;
  /** Get and set the voxel size. */
  std::string patchSizeStr;
  this->GetConfiguration()->ReadParameter(patchSizeStr, type + "PatchSize", this->GetComponentLabel(), level, 0);
  patchSizeVec = this->GetVectorFromString<std::string>(modelsPathVec.size(), patchSizeStr, "5*5*5");

  /** Get and set the voxel size. */
  std::string voxelSizeStr;
  this->GetConfiguration()->ReadParameter(voxelSizeStr, type + "VoxelSize", this->GetComponentLabel(), level, 0);
  std::vector<std::string> voxelSizeVec =
    this->GetVectorFromString<std::string>(modelsPathVec.size(), voxelSizeStr, "1.5*1.5*1.5");

  /** Get and set the Strides. */
  std::string layersMaskStr;
  this->GetConfiguration()->ReadParameter(layersMaskStr, type + "LayersMask", this->GetComponentLabel(), level, 0);
  std::vector<std::string> layersMaskVec =
    this->GetVectorFromString<std::string>(modelsPathVec.size(), layersMaskStr, "1");

  // Build the ModelConfiguration object for each model.
  // Each configuration includes model path, input dimension, channel count,
  // patch size, voxel size, and layer mask.
  // In static mode, we flag the model to cache features at init.
  for (int i = 0; i < modelsPathVec.size(); i++)
  {
    try
    {
      if (mode == "Static")
      {
        modelsConfiguration.emplace_back(
          modelsPathVec[i],
          modelsDimensionVec[i],
          numberOfChannelsVec[i],
          this->GetVectorFromString<long>(modelsDimensionVec[i], patchSizeVec[i], 5, '*'),
          this->GetVectorFromString<float>(imageDimension, voxelSizeVec[i], 1.5, '*'),
          this->GetVectorFromString<bool>(layersMaskVec[i], true, '\0'),
          true);
      }
      else
      {
        modelsConfiguration.emplace_back(
          modelsPathVec[i],
          modelsDimensionVec[i],
          numberOfChannelsVec[i],
          this->GetVectorFromString<long>(modelsDimensionVec[i], patchSizeVec[i], 5, '*'),
          this->GetVectorFromString<float>(modelsDimensionVec[i], voxelSizeVec[i], 1.5, '*'),
          this->GetVectorFromString<bool>(layersMaskVec[i], true, '\0'),
          false);
      }
    }
    catch (const c10::Error & e)
    {
      itkExceptionMacro("ERROR: the fixed model are not loaded from this file : " << modelsPathVec[i] << ".");
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

  // Read the mode of operation for the metric: "Jacobian" or "Static".
  // - Static: features are precomputed and optionally saved.
  // - Jacobian: gradients are propagated through the models.
  std::string mode = "Jacobian";
  this->GetConfiguration()->ReadParameter(mode, "Mode", this->GetComponentLabel(), level, 0);
  if (mode != "Jacobian" && mode != "Static")
  {
    itkExceptionMacro(
      "Invalid mode: '" << mode << "'. Supported modes are 'Jacobian' and 'Static'. Please check the configuration.");
  }
  this->SetMode(mode);

  // Try to read shared model path (used if FixedModelsPath / MovingModelsPath are not provided)
  std::string modelsPath;
  bool        hasSharedModel =
    this->GetConfiguration()->ReadParameter(modelsPath, "ModelsPath", this->GetComponentLabel(), level, 0);

  // Generate configuration for fixed and moving images.
  // Priority: FixedModelsPath > MovingModelsPath > ModelsPath
  // This allows shared or asymmetric feature extractors.

  // Generate fixed model configuration (fallback to shared if FixedModelsPath is missing)
  bool hasFixed =
    this->GetConfiguration()->ReadParameter(modelsPath, "FixedModelsPath", this->GetComponentLabel(), level, 0);
  if (hasFixed)
  {
    this->SetFixedModelsConfiguration(this->GenerateModelsConfiguration(level, "Fixed", mode, FixedImageDimension));
  }
  else if (hasSharedModel)
  {
    this->SetFixedModelsConfiguration(this->GenerateModelsConfiguration(level, "", mode, FixedImageDimension));
  }
  else
  {
    itkExceptionMacro("Missing parameter: FixedModelsPath or shared ModelsPath must be provided.");
  }

  // Generate moving model configuration (fallback to shared if MovingModelsPath is missing)
  bool hasMoving =
    this->GetConfiguration()->ReadParameter(modelsPath, "MovingModelsPath", this->GetComponentLabel(), level, 0);
  if (hasMoving)
  {
    this->SetMovingModelsConfiguration(this->GenerateModelsConfiguration(level, "Moving", mode, MovingImageDimension));
  }
  else if (hasSharedModel)
  {
    this->SetMovingModelsConfiguration(this->GenerateModelsConfiguration(level, "", mode, MovingImageDimension));
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
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); i++)
  {
    if (this->GetFixedModelsConfiguration()[i].m_dimension > FixedImageDimension)
    {
      itkExceptionMacro("ERROR: The dimension of the fixed input model image exceeds the allowed image dimensions. "
                        "Expected a maximum of "
                        << FixedImageDimension << " dimension(s), but received "
                        << this->GetFixedModelsConfiguration()[i].m_dimension << " dimension(s) for model index " << i
                        << ". "
                           "Please verify the input model dimensions.");
    }
  }

  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); i++)
  {
    if (this->GetMovingModelsConfiguration()[i].m_dimension > FixedImageDimension)
    {
      itkExceptionMacro("ERROR: The dimension of the moving input model image exceeds the allowed image dimensions. "
                        "Expected a maximum of "
                        << MovingImageDimension << " dimension(s), but received "
                        << this->GetMovingModelsConfiguration()[i].m_dimension << " dimension(s) for model index " << i
                        << ". "
                           "Please verify the input model dimensions.");
    }
  }

  // Choose GPU device if available and requested, fallback to CPU otherwise.
  // Raise explicit errors if user-requested GPU index is invalid.
  int device = 0;
  this->GetConfiguration()->ReadParameter(device, "GPU", this->GetComponentLabel(), level, 0);

  // Select computation device (GPU or CPU) based on availability and config
  if (device >= 0)
  {
    if (torch::cuda::is_available())
    {
      int availableGPUs = torch::cuda::device_count();
      if (device < availableGPUs)
      {
        this->SetGPU(torch::Device(torch::kCUDA, device));
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
    this->SetGPU(torch::Device(torch::kCPU));
  }

  // Handle feature map export setup.
  // If WriteFeatureMaps is set (either "true" or a path), create the directory.
  // Store the path to be reused during feature writing.
  if (mode == "Static")
  {
    int featuresMapUpdateInterval = -1;
    this->GetConfiguration()->ReadParameter(
      featuresMapUpdateInterval, "FeaturesMapUpdateInterval", this->GetComponentLabel(), level, 0);
    this->SetFeaturesMapUpdateInterval(featuresMapUpdateInterval);
  }

  //
  int fixedNumberOfLayers = 0;
  for (int i = 0; i < this->GetFixedModelsConfiguration().size(); i++)
  {
    std::vector<bool> layersMask = this->GetFixedModelsConfiguration()[i].m_layersMask;
    fixedNumberOfLayers += std::count(layersMask.begin(), layersMask.end(), true);
    this->GetFixedModelsConfiguration()[i].m_model->to(this->GetGPU());
  }

  int movingNumberOfLayers = 0;
  for (int i = 0; i < this->GetMovingModelsConfiguration().size(); i++)
  {
    std::vector<bool> layersMask = this->GetMovingModelsConfiguration()[i].m_layersMask;
    movingNumberOfLayers += std::count(layersMask.begin(), layersMask.end(), true);
    this->GetMovingModelsConfiguration()[i].m_model->to(this->GetGPU());
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
  this->GetConfiguration()->ReadParameter(subsetFeaturesStr, "SubsetFeatures", this->GetComponentLabel(), level, 0);
  this->SetSubsetFeatures(this->GetVectorFromString<unsigned int>(fixedNumberOfLayers, subsetFeaturesStr, 32));

  /** Get and set the SubsetFeatures. */
  std::string pcaStr;
  this->GetConfiguration()->ReadParameter(pcaStr, "PCA", this->GetComponentLabel(), level, 0);
  this->SetPCA(this->GetVectorFromString<unsigned int>(fixedNumberOfLayers, pcaStr, 0));

  /** Get and set the LayersWeight. */
  std::string layersWeightStr;
  this->GetConfiguration()->ReadParameter(layersWeightStr, "LayersWeight", this->GetComponentLabel(), level, 0);
  this->SetLayersWeight(this->GetVectorFromString<float>(fixedNumberOfLayers, layersWeightStr, 1.0));
  this->SetWriteFeatureMaps(false);

  if (mode == "Static")
  {
    std::string writeFeatureMapsStr;
    this->GetConfiguration()->ReadParameter(
      writeFeatureMapsStr, "WriteFeatureMaps", this->GetComponentLabel(), level, 0);
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
  this->GetConfiguration()->ReadParameter(distanceStr, "Distance", this->GetComponentLabel(), level, 0);
  this->SetDistance(this->GetVectorFromString<std::string>(fixedNumberOfLayers, distanceStr, "L2"));
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
