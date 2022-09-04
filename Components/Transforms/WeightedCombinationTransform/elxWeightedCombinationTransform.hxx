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

#ifndef elxWeightedCombinationTransform_hxx
#define elxWeightedCombinationTransform_hxx

#include "elxWeightedCombinationTransform.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
WeightedCombinationTransformElastix<TElastix>::WeightedCombinationTransformElastix()
{
  this->SetCurrentTransform(this->m_WeightedCombinationTransform);
} // end Constructor


/*
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
WeightedCombinationTransformElastix<TElastix>::BeforeRegistration()
{
  /** Set the normalizedWeights parameter. It must be correct in order to set the scales properly.
   * \todo: this parameter may change each resolution. */
  bool normalizeWeights = false;
  this->m_Configuration->ReadParameter(normalizeWeights, "NormalizeCombinationWeights", 0);
  this->m_WeightedCombinationTransform->SetNormalizeWeights(normalizeWeights);

  /** Give initial parameters to this->m_Registration.*/
  this->InitializeTransform();

  /** Set the scales. */
  this->SetScales();

} // end BeforeRegistration


/**
 * ************************* InitializeTransform *********************
 * Initialize transform to prepare it for registration.
 */

template <class TElastix>
void
WeightedCombinationTransformElastix<TElastix>::InitializeTransform()
{
  /** Load subtransforms specified in parameter file. */
  this->LoadSubTransforms();

  /** Some helper variables */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  const double                 Nd = static_cast<double>(numberOfParameters);

  /** Equal weights */
  ParametersType parameters(numberOfParameters);
  if (this->m_WeightedCombinationTransform->GetNormalizeWeights())
  {
    parameters.Fill(1.0 / Nd);
  }
  else
  {
    parameters.Fill(0.0);
  }
  this->m_WeightedCombinationTransform->SetParameters(parameters);

  /** Set the initial parameters in this->m_Registration.*/
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

} // end InitializeTransform


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
WeightedCombinationTransformElastix<TElastix>::ReadFromFile()
{
  /** Load subtransforms specified in transform parameter file. */
  this->LoadSubTransforms();

  /** Set the normalizeWeights option */
  bool normalizeWeights = false;
  this->m_Configuration->ReadParameter(normalizeWeights, "NormalizeCombinationWeights", 0);
  this->m_WeightedCombinationTransform->SetNormalizeWeights(normalizeWeights);

  /** Call the ReadFromFile from the TransformBase to read in the parameters.  */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CustomizeTransformParametersMap ************************
 */

template <class TElastix>
auto
WeightedCombinationTransformElastix<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_WeightedCombinationTransform;

  return { { "NormalizeCombinationWeights", { Conversion::ToString(itkTransform.GetNormalizeWeights()) } },
           { "SubTransforms", m_SubTransformFileNames } };

} // end CustomizeTransformParametersMap()


/**
 * ************************* SetScales *********************
 */

template <class TElastix>
void
WeightedCombinationTransformElastix<TElastix>::SetScales()
{
  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters);
  newscales.Fill(1.0);

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter(automaticScalesEstimation, "AutomaticScalesEstimation", 0, false);

  if (automaticScalesEstimation)
  {
    elxout << "Scales are estimated automatically." << std::endl;
    this->AutomaticScalesEstimation(newscales);
  }
  else
  {
    const std::size_t count = this->m_Configuration->CountNumberOfParameterEntries("Scales");

    if (count == numberOfParameters)
    {
      /** Read the user-supplied values/ */
      std::vector<double> newscalesvec(numberOfParameters);
      this->m_Configuration->ReadParameter(newscalesvec, "Scales", 0, numberOfParameters - 1, true);
      for (unsigned int i = 0; i < numberOfParameters; ++i)
      {
        newscales[i] = newscalesvec[i];
      }
    }
    else if (count != 0)
    {
      /** In this case an error is made in the parameter-file.
       * An error is thrown, because using erroneous scales in the optimizer
       * can give unpredictable results.
       */
      itkExceptionMacro(<< "ERROR: The Scales-option in the parameter-file has not been set properly.");
    }

  } // end else: no automaticScalesEstimation

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** And set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ************************* LoadSubTransforms *********************
 */

template <class TElastix>
void
WeightedCombinationTransformElastix<TElastix>::LoadSubTransforms()
{
  /** Typedef's from ComponentDatabase. */
  using ComponentDescriptionType = typename Superclass2::ComponentDescriptionType;
  using PtrToCreator = typename Superclass2::PtrToCreator;

  const std::size_t N = this->m_Configuration->CountNumberOfParameterEntries("SubTransforms");

  if (N == 0)
  {
    itkExceptionMacro(<< "ERROR: At least one SubTransform should be specified.");
  }
  else
  {
    this->m_SubTransformFileNames.resize(N);
    this->m_Configuration->ReadParameter(this->m_SubTransformFileNames, "SubTransforms", 0, N - 1, true);
  }

  /** Create a vector of subTransform pointers and initialize to null pointers.
   * Note that std::vector will properly initialize its elements to null (by default).
   * \todo: make it a member variable if it appears to needed later */
  TransformContainerType subTransforms(N);

  /** Load each subTransform */
  for (unsigned int i = 0; i < N; ++i)
  {
    /** \todo: large parts of these code were copied from the elx::TransformBase.
     * Could we put some functionality in a function? */

    /** Read the name of the subTransform */
    const std::string & subTransformFileName = this->m_SubTransformFileNames[i];

    /** Create a new configuration, which will be initialized with
     * the subtransformFileName. */
    const auto configurationSubTransform = Configuration::New();

    /** Create argmapInitialTransform. */
    CommandLineArgumentMapType argmapSubTransform;
    argmapSubTransform.insert(CommandLineEntryType("-tp", subTransformFileName));

    int initfailure = configurationSubTransform->Initialize(argmapSubTransform);
    if (initfailure != 0)
    {
      itkExceptionMacro(<< "ERROR: Reading SubTransform parameters failed: " << subTransformFileName);
    }

    /** Read the SubTransform name. */
    ComponentDescriptionType subTransformName = "AffineTransform";
    configurationSubTransform->ReadParameter(subTransformName, "Transform", 0);

    /** Create a SubTransform. */
    const PtrToCreator creator =
      ElastixMain::GetComponentDatabase().GetCreator(subTransformName, this->m_Elastix->GetDBIndex());
    const itk::Object::Pointer subTransform = (creator == nullptr) ? nullptr : creator();

    /** Cast to TransformBase */
    Superclass2 * elx_subTransform = dynamic_cast<Superclass2 *>(subTransform.GetPointer());

    /** Call the ReadFromFile method of the elx_subTransform. */
    if (elx_subTransform)
    {
      elx_subTransform->SetElastix(this->GetElastix());
      elx_subTransform->SetConfiguration(configurationSubTransform);
      elx_subTransform->ReadFromFile();

      /** Set in vector of subTransforms. */
      SubTransformType * testPointer = dynamic_cast<SubTransformType *>(subTransform.GetPointer());
      subTransforms[i] = testPointer;
    }

    /** Check if no errors occured: */
    if (subTransforms[i].IsNull())
    {
      xl::xout["error"] << "ERROR: Error while trying to load the SubTransform " << subTransformFileName << std::endl;
      itkExceptionMacro(<< "ERROR: Loading SubTransforms failed!");
    }

  } // end for loop over subTransforms

  /** Set the subTransforms in the WeightedCombination object. */
  this->m_WeightedCombinationTransform->SetTransformContainer(subTransforms);

} // end LoadSubTransforms()


} // end namespace elastix

#endif // end #ifndef elxWeightedCombinationTransform_hxx
