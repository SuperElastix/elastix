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
#ifndef itkMultiBSplineDeformableTransformWithNormal_hxx
#define itkMultiBSplineDeformableTransformWithNormal_hxx

#include "itkMultiBSplineDeformableTransformWithNormal.h"
#include "itkStatisticsImageFilter.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkConstantPadImageFilter.h"

namespace itk
{

// Constructor with default arguments
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::
  MultiBSplineDeformableTransformWithNormal()
  : Superclass(SpaceDimension)
{
  // By default this class handle a unique Transform
  this->m_NbLabels = 0;
  this->m_Labels = nullptr;
  this->m_LabelsInterpolator = nullptr;
  this->m_Trans.resize(1);
  // keep transform 0 to store parameters that are not kept here (GridSize, ...)
  this->m_Trans[0] = TransformType::New();
  this->m_Para.resize(0);
  this->m_LastJacobian = -1;
  this->m_LocalBases = ImageBaseType::New();

  this->m_InternalParametersBuffer = ParametersType(0);
  // Make sure the parameters pointer is not NULL after construction.
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);
}


// Get the number of parameters
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetNumberOfParameters() const
  -> NumberOfParametersType
{
  if (m_NbLabels > 0)
  {
    return (1 + (SpaceDimension - 1) * m_NbLabels) * m_Trans[0]->GetNumberOfParametersPerDimension();
  }
  else
  {
    return 0;
  }
}


// Get the number of parameters per dimension
// FIXME :  Do we need to declare this function ?
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetNumberOfParametersPerDimension()
  const -> NumberOfParametersType
{
  // FIXME : Depends on which dimension we are speaking here. should check it
  if (m_NbLabels > 0)
  {
    return m_Trans[0]->GetNumberOfParametersPerDimension();
  }
  else
  {
    return 0;
  }
}


#define LOOP_ON_LABELS(FUNC, ARGS)                                                                                     \
  for (unsigned i = 0; i <= m_NbLabels; ++i)                                                                           \
  {                                                                                                                    \
    m_Trans[i]->FUNC(ARGS);                                                                                            \
  }

#define SET_ALL_LABELS(FUNC, TYPE)                                                                                     \
  template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>                                    \
  void MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::Set##FUNC(const TYPE _arg)   \
  {                                                                                                                    \
    if (_arg != this->Get##FUNC())                                                                                     \
    {                                                                                                                  \
      LOOP_ON_LABELS(Set##FUNC, _arg);                                                                                 \
      this->Modified();                                                                                                \
    }                                                                                                                  \
  }

#define GET_FIRST_LABEL(FUNC, TYPE)                                                                                    \
  template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>                                    \
  auto MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::Get##FUNC() const->TYPE      \
  {                                                                                                                    \
    return m_Trans[0]->Get##FUNC();                                                                                    \
  }

// Set the grid region
SET_ALL_LABELS(GridRegion, RegionType &);

// Get the grid region
GET_FIRST_LABEL(GridRegion, RegionType);

// Set the grid spacing
SET_ALL_LABELS(GridSpacing, SpacingType &);

// Get the grid spacing
GET_FIRST_LABEL(GridSpacing, SpacingType);

// Set the grid direction
SET_ALL_LABELS(GridDirection, DirectionType &);

// Get the grid direction
GET_FIRST_LABEL(GridDirection, DirectionType);

// Set the grid origin
SET_ALL_LABELS(GridOrigin, OriginType &);

// Get the grid origin
GET_FIRST_LABEL(GridOrigin, OriginType);

#undef SET_ALL_LABELS
#undef GET_FIRST_LABEL

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::SetLabels(ImageLabelType * labels)
{
  using StatisticsType = StatisticsImageFilter<ImageLabelType>;
  if (labels != this->m_Labels)
  {
    // Save current settings
    this->m_Labels = labels;
    ParametersType para = this->GetFixedParameters();
    auto           stat = StatisticsType::New();
    stat->SetInput(this->m_Labels);
    stat->Update();
    this->m_NbLabels = stat->GetMaximum() + 1;
    this->m_Trans.resize(this->m_NbLabels + 1);
    this->m_Para.resize(this->m_NbLabels + 1);
    for (unsigned i = 0; i <= this->m_NbLabels; ++i)
    {
      this->m_Trans[i] = TransformType::New();
    }
    this->m_LabelsInterpolator = ImageLabelInterpolator::New();
    this->m_LabelsInterpolator->SetInputImage(this->m_Labels);
    // Restore settings
    this->SetFixedParameters(para);
  }
}


template <class TScalarType, unsigned int NDimensions>
struct UpdateLocalBases_impl
{
  using VectorType = itk::Vector<TScalarType, NDimensions>;
  using BaseType = itk::Vector<VectorType, NDimensions>;
  using ImageVectorType = itk::Image<VectorType, NDimensions>;
  using ImageVectorPointer = typename ImageVectorType::Pointer;
  using ImageBaseType = itk::Image<BaseType, NDimensions>;
  using ImageBasePointer = typename ImageBaseType::Pointer;

  static void
  Do(ImageBaseType *, ImageVectorType *)
  {
    itkGenericExceptionMacro(<< "MultiBSplineDeformableTransformWithNormal only works with 3D image for the moment");
  }
};

template <class TScalarType>
struct UpdateLocalBases_impl<TScalarType, 2>
{
  static const unsigned NDimensions = 2;
  using VectorType = itk::Vector<TScalarType, NDimensions>;
  using BaseType = itk::Vector<VectorType, NDimensions>;
  using ImageVectorType = itk::Image<VectorType, NDimensions>;
  using ImageVectorPointer = typename ImageVectorType::Pointer;
  using ImageBaseType = itk::Image<BaseType, NDimensions>;
  using ImageBasePointer = typename ImageBaseType::Pointer;

  static void
  Do(ImageBaseType * bases, ImageVectorType * normals)
  {
    const TScalarType base_x[] = { 1, 0 };
    const TScalarType base_y[] = { 0, 1 };

    using ImageVectorInterpolator = itk::NearestNeighborInterpolateImageFunction<ImageVectorType, TScalarType>;
    using ImageVectorInterpolatorPointer = typename ImageVectorInterpolator::Pointer;
    ImageVectorInterpolatorPointer vinterp = ImageVectorInterpolator::New();
    vinterp->SetInputImage(normals);

    using IteratorType = ImageRegionIterator<ImageBaseType>;
    IteratorType it(bases, bases->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      BaseType                          b;
      typename ImageBaseType::PointType p;
      bases->TransformIndexToPhysicalPoint(it.GetIndex(), p);
      typename ImageBaseType::IndexType idx;
      vinterp->ConvertPointToNearestIndex(p, idx);
      if (!vinterp->IsInsideBuffer(idx))
      {
        // far from an interface keep (x,y,z) base
        b[0] = VectorType(base_x);
        b[1] = VectorType(base_y);
        it.Set(b);
        continue;
      }

      VectorType n = vinterp->EvaluateAtIndex(idx);
      n.Normalize();
      if (n.GetNorm() < 0.1)
      {
        std::cout << "Should never append" << std::endl;
        // far from an interface keep (x,y,z) base
        b[0] = VectorType(base_x);
        b[1] = VectorType(base_y);
        it.Set(b);
        continue;
      }

      b[0] = n;
      b[1][0] = n[1];
      b[1][1] = -n[0];
      it.Set(b);
    }
  }
};

template <class TScalarType>
struct UpdateLocalBases_impl<TScalarType, 3>
{
  static const unsigned NDimensions = 3;
  using VectorType = itk::Vector<TScalarType, NDimensions>;
  using BaseType = itk::Vector<VectorType, NDimensions>;
  using ImageVectorType = itk::Image<VectorType, NDimensions>;
  using ImageVectorPointer = typename ImageVectorType::Pointer;
  using ImageBaseType = itk::Image<BaseType, NDimensions>;
  using ImageBasePointer = typename ImageBaseType::Pointer;

  static void
  Do(ImageBaseType * bases, ImageVectorType * normals)
  {
    const TScalarType base_x[] = { 1, 0, 0 };
    const TScalarType base_y[] = { 0, 1, 0 };
    const TScalarType base_z[] = { 0, 0, 1 };

    using ImageVectorInterpolator = itk::NearestNeighborInterpolateImageFunction<ImageVectorType, TScalarType>;
    using ImageVectorInterpolatorPointer = typename ImageVectorInterpolator::Pointer;
    ImageVectorInterpolatorPointer vinterp = ImageVectorInterpolator::New();
    vinterp->SetInputImage(normals);

    using IteratorType = ImageRegionIterator<ImageBaseType>;
    IteratorType it(bases, bases->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      BaseType                          b;
      typename ImageBaseType::PointType p;
      bases->TransformIndexToPhysicalPoint(it.GetIndex(), p);
      typename ImageBaseType::IndexType idx;
      vinterp->ConvertPointToNearestIndex(p, idx);
      if (!vinterp->IsInsideBuffer(idx))
      {
        // far from an interface keep (x,y,z) base
        b[0] = VectorType(base_x);
        b[1] = VectorType(base_y);
        b[2] = VectorType(base_z);
        it.Set(b);
        continue;
      }

      VectorType n = vinterp->EvaluateAtIndex(idx);
      n.Normalize();
      if (n.GetNorm() < 0.1)
      {
        std::cout << "Should never append" << std::endl;
        // far from an interface keep (x,y,z) base
        b[0] = VectorType(base_x);
        b[1] = VectorType(base_y);
        b[2] = VectorType(base_z);
        it.Set(b);
        continue;
      }

      b[0] = n;

      // find the must non colinear to vector wrt n
      VectorType tmp;
      if (std::abs(n[0]) < std::abs(n[1]))
      {
        if (std::abs(n[0]) < std::abs(n[2]))
        {
          tmp = base_x;
        }
        else
        {
          tmp = base_z;
        }
      }
      else
      {
        if (std::abs(n[1]) < std::abs(n[2]))
        {
          tmp = base_y;
        }
        else
        {
          tmp = base_z;
        }
      }

      // find u and v in order to form a local orthonormal base with n
      tmp = CrossProduct(n, tmp);
      tmp.Normalize();
      b[1] = tmp;
      tmp = CrossProduct(n, tmp);
      tmp.Normalize();
      b[2] = tmp;
      it.Set(b);
    }
  }
};

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::UpdateLocalBases()
{
  using ImageDoubleType = itk::Image<double, Self::SpaceDimension>;
  using PadFilterType = itk::ConstantPadImageFilter<ImageLabelType, ImageLabelType>;
  using DistFilterType = itk::ApproximateSignedDistanceMapImageFilter<ImageLabelType, ImageDoubleType>;
  using SmoothFilterType = itk::SmoothingRecursiveGaussianImageFilter<ImageDoubleType, ImageDoubleType>;
  using GradFilterType = itk::GradientImageFilter<ImageDoubleType, double, double>;
  using LabelExtractorType = itk::BinaryThresholdImageFilter<ImageLabelType, ImageLabelType>;
  using AddVectorImageType = itk::AddImageFilter<ImageVectorType, ImageVectorType, ImageVectorType>;
  using MaskVectorImageType = itk::MaskImageFilter<ImageVectorType, ImageLabelType, ImageVectorType>;
  using PointType = typename ImageLabelType::PointType;
  using RegionType = typename ImageLabelType::RegionType;
  using SpacingType = typename ImageLabelType::SpacingType;

  PointType transOrig = GetGridOrigin();
  PointType transEnd;
  for (unsigned i = 0; i < NDimensions; ++i)
  {
    transEnd[i] = transOrig[i] + (GetGridRegion().GetSize()[i] - GetGridRegion().GetIndex()[i]) * GetGridSpacing()[i];
  }

  PointType   labelOrig = this->m_Labels->GetOrigin();
  RegionType  labelReg = this->m_Labels->GetLargestPossibleRegion();
  SpacingType labelSpac = this->m_Labels->GetSpacing();
  PointType   labelEnd;
  for (unsigned i = 0; i < NDimensions; ++i)
  {
    labelEnd[i] = labelOrig[i] + (labelReg.GetSize()[i] - labelReg.GetIndex()[i]) * labelSpac[i];
  }

  typename ImageLabelType::SizeType lowerExtend;
  for (unsigned i = 0; i < NDimensions; ++i)
  {
    lowerExtend[i] = std::ceil((labelOrig[i] - transOrig[i]) / labelSpac[i]);
  }

  typename ImageLabelType::SizeType upperExtend;
  for (unsigned i = 0; i < NDimensions; ++i)
  {
    upperExtend[i] = std::ceil((transEnd[i] - labelEnd[i])) / labelSpac[i];
  }

  auto padFilter = PadFilterType::New();
  padFilter->SetInput(this->m_Labels);
  padFilter->SetPadLowerBound(lowerExtend);
  padFilter->SetPadUpperBound(upperExtend);
  padFilter->SetConstant(0);

  for (int l = 0; l < this->m_NbLabels; ++l)
  {
    auto labelExtractor = LabelExtractorType::New();
    labelExtractor->SetInput(padFilter->GetOutput());
    labelExtractor->SetLowerThreshold(l);
    labelExtractor->SetUpperThreshold(l);
    labelExtractor->SetInsideValue(1);
    labelExtractor->SetOutsideValue(0);

    auto distFilter = DistFilterType::New();
    distFilter->SetInsideValue(1);
    distFilter->SetOutsideValue(0);
    distFilter->SetInput(labelExtractor->GetOutput());

    auto smoothFilter = SmoothFilterType::New();
    smoothFilter->SetInput(distFilter->GetOutput());
    smoothFilter->SetSigma(4.);

    auto gradFilter = GradFilterType::New();
    gradFilter->SetInput(smoothFilter->GetOutput());

    const auto castFilter = itk::CastImageFilter<typename GradFilterType::OutputImageType, ImageVectorType>::New();

    castFilter->SetInput(gradFilter->GetOutput());

    auto maskFilter = MaskVectorImageType::New();
    maskFilter->SetInput(castFilter->GetOutput());
    maskFilter->SetMaskImage(labelExtractor->GetOutput());
    maskFilter->SetOutsideValue(itk::NumericTraits<VectorType>::ZeroValue());
    maskFilter->Update();

    if (l == 0)
    {
      this->m_LabelsNormals = maskFilter->GetOutput();
    }
    else
    {
      auto addFilter = AddVectorImageType::New();
      addFilter->SetInput1(this->m_LabelsNormals);
      addFilter->SetInput2(maskFilter->GetOutput());
      addFilter->Update();
      this->m_LabelsNormals = addFilter->GetOutput();
    }
  }

  m_LocalBases = ImageBaseType::New();
  m_LocalBases->SetRegions(GetGridRegion());
  m_LocalBases->SetSpacing(GetGridSpacing());
  m_LocalBases->SetOrigin(GetGridOrigin());
  m_LocalBases->SetDirection(GetGridDirection());
  m_LocalBases->Allocate();
  UpdateLocalBases_impl<TScalarType, NDimensions>::Do(this->m_LocalBases, this->m_LabelsNormals);
}


// Set the parameters
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::SetIdentity()
{
  LOOP_ON_LABELS(SetIdentity, );
  if (this->m_InputParametersPointer)
  {
    ParametersType * parameters = const_cast<ParametersType *>(this->m_InputParametersPointer);
    parameters->Fill(0.0);
    this->Modified();
  }
  else
  {
    itkExceptionMacro(<< "Input parameters for the spline haven't been set ! Set them using the SetParameters or "
                         "SetCoefficientImage method first.");
  }
}


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::DispatchParameters(
  const ParametersType & parameters)
{
  for (unsigned i = 0; i <= m_NbLabels; ++i)
  {
    m_Para[i].SetSize(m_Trans[i]->GetNumberOfParameters());
  }

  using BaseContainer = typename ImageBaseType::PixelContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();
  unsigned              ParametersPerDimension = m_Trans[0]->GetNumberOfParametersPerDimension();
  for (unsigned i = 0; i < ParametersPerDimension; ++i)
  {
    VectorType tmp = bases[i][0] * parameters.GetElement(i);
    for (unsigned d = 0; d < Self::SpaceDimension; ++d)
    {
      m_Para[0].SetElement(i + d * ParametersPerDimension, tmp[d]);
    }

    for (unsigned l = 1; l <= m_NbLabels; ++l)
    {
      for (unsigned d = 0; d < Self::SpaceDimension; ++d)
      {
        tmp[d] = 0;
      }

      for (unsigned d = 1; d < Self::SpaceDimension; ++d)
      {
        tmp +=
          bases[i][d] * parameters.GetElement(i + ((Self::SpaceDimension - 1) * (l - 1) + d) * ParametersPerDimension);
      }

      for (unsigned d = 0; d < Self::SpaceDimension; ++d)
      {
        m_Para[l].SetElement(i + d * ParametersPerDimension, tmp[d]);
      }
    }
  }
  for (unsigned i = 0; i <= m_NbLabels; ++i)
  {
    m_Trans[i]->SetParameters(m_Para[i]);
  }
}


// Set the parameters
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::SetParameters(
  const ParametersType & parameters)
{
  // check if the number of parameters match the
  // expected number of parameters
  if (parameters.Size() != this->GetNumberOfParameters())
  {
    itkExceptionMacro(<< "Mismatched between parameters size " << parameters.size() << " and region size "
                      << this->GetNumberOfParameters());
  }

  // Clean up buffered parameters
  this->m_InternalParametersBuffer = ParametersType(0);

  // Keep a reference to the input parameters
  this->m_InputParametersPointer = &parameters;

  DispatchParameters(parameters);

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}


// Set the Fixed Parameters
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::SetFixedParameters(
  const ParametersType & passedParameters)
{
  LOOP_ON_LABELS(SetFixedParameters, passedParameters);
  this->Modified();
}


// Set the parameters by value
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::SetParametersByValue(
  const ParametersType & parameters)
{
  // check if the number of parameters match the
  // expected number of parameters
  if (parameters.Size() != this->GetNumberOfParameters())
  {
    itkExceptionMacro(<< "Mismatched between parameters size " << parameters.size() << " and region size "
                      << this->GetNumberOfParameters());
  }

  // copy it
  this->m_InternalParametersBuffer = parameters;
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);

  DispatchParameters(parameters);

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}


// Get the parameters
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetParameters() const
  -> const ParametersType &
{
  /** NOTE: For efficiency, this class does not keep a copy of the parameters -
   * it just keeps pointer to input parameters.
   */
  if (nullptr == this->m_InputParametersPointer)
  {
    itkExceptionMacro(<< "Cannot GetParameters() because m_InputParametersPointer is NULL. Perhaps "
                         "SetCoefficientImages() has been called causing the NULL pointer.");
  }

  return (*this->m_InputParametersPointer);
}

// Get the parameters
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetFixedParameters() const
  -> const ParametersType &
{
  return (m_Trans[0]->GetFixedParameters());
}

// Print self
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::PrintSelf(std::ostream & os,
                                                                                             Indent indent) const
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "NbLabels : " << m_NbLabels << std::endl;
  itk::Indent ind = indent.GetNextIndent();

  os << indent << "Normal " << std::endl;
  m_Trans[0]->Print(os, ind);
  for (unsigned i = 1; i <= m_NbLabels; ++i)
  {
    os << indent << "Label " << i << std::endl;
    m_Trans[i]->Print(os, ind);
  }
}


#undef LOOP_ON_LABELS

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
inline void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::PointToLabel(
  const InputPointType & p,
  int &                  l) const
{
  l = 0;
  assert(this->m_Labels);
  typename ImageLabelInterpolator::IndexType idx;
  this->m_LabelsInterpolator->ConvertPointToNearestIndex(p, idx);
  if (this->m_LabelsInterpolator->IsInsideBuffer(idx))
  {
    l = static_cast<int>(this->m_LabelsInterpolator->EvaluateAtIndex(idx)) + 1;
  }
}


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
auto
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::TransformPoint(
  const InputPointType & point) const -> OutputPointType
{
  int lidx = 0;
  this->PointToLabel(point, lidx);
  if (lidx == 0)
  {
    return point;
  }

  OutputPointType res = m_Trans[0]->TransformPoint(point) + (m_Trans[lidx]->TransformPoint(point) - point);
  return res;
}


// template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
// auto
// MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>
//::GetJacobian( const InputPointType & point ) const -> const JacobianType&
//{
//  this->m_Jacobian.set_size(SpaceDimension, this->GetNumberOfParameters());
//  this->m_Jacobian.Fill(0.0);
//  JacobianType jacobian;
//  NonZeroJacobianIndicesType nonZeroJacobianIndices;
//  this->GetJacobian(point, jacobian, nonZeroJacobianIndices);
//  for (unsigned i = 0; i < nonZeroJacobianIndices.size(); ++i)
//    for (unsigned j = 0; j < SpaceDimension; ++j)
//    this->m_Jacobian[j][nonZeroJacobianIndices[i]] = jacobian[j][i];
//  return this->m_Jacobian;
//}

/**
 * ********************* GetJacobian ****************************
 */

template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetJacobian(
  const InputPointType &       inputPoint,
  JacobianType &               jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  if (this->GetNumberOfParameters() == 0)
  {
    jacobian.SetSize(SpaceDimension, 0);
    nonZeroJacobianIndices.resize(0);
    return;
  }

  // Initialize
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if ((jacobian.cols() != nnzji) || (jacobian.rows() != SpaceDimension))
  {
    jacobian.SetSize(SpaceDimension, nnzji);
  }
  jacobian.Fill(0.0);

  // This implements a sparse version of the Jacobian.
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  int lidx = 0;
  PointToLabel(inputPoint, lidx);

  if (lidx == 0)
  {
    // Return some dummy
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (unsigned int i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  JacobianType njac, ljac;
  njac.SetSize(SpaceDimension, nnzji);
  ljac.SetSize(SpaceDimension, nnzji);

  // nzji should be the same so keep only one
  m_Trans[0]->GetJacobian(inputPoint, njac, nonZeroJacobianIndices);
  m_Trans[lidx]->GetJacobian(inputPoint, ljac, nonZeroJacobianIndices);

  // Convert the physical point to a continuous index, which
  // is needed for the 'Evaluate()' functions below.
  const typename TransformType::ContinuousIndexType cindex =
    m_Trans[lidx]->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero Jacobian
  if (!m_Trans[lidx]->InsideValidRegion(cindex))
  {
    // Return some dummy
    nonZeroJacobianIndices.resize(m_Trans[lidx]->GetNumberOfNonZeroJacobianIndices());
    for (unsigned int i = 0; i < m_Trans[lidx]->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  using BaseContainer = typename ImageBaseType::PixelContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();

  const unsigned nweights = this->GetNumberOfWeights();
  for (unsigned i = 0; i < nweights; ++i)
  {
    VectorType tmp = bases[nonZeroJacobianIndices[i]][0];
    for (unsigned j = 0; j < SpaceDimension; ++j)
    {
      jacobian[j][i] = tmp[j] * njac[j][i + j * nweights];
    }

    for (unsigned d = 1; d < SpaceDimension; ++d)
    {
      tmp = bases[nonZeroJacobianIndices[i]][d];
      for (unsigned j = 0; j < SpaceDimension; ++j)
      {
        jacobian[j][i + d * nweights] = tmp[j] * ljac[j][i + j * nweights];
      }
    }
  }

  // move non zero indices to match label positions
  if (lidx > 1)
  {
    unsigned to_add = (lidx - 1) * m_Trans[0]->GetNumberOfParametersPerDimension() * (SpaceDimension - 1);
    for (unsigned i = 0; i < nweights; ++i)
    {
      for (unsigned d = 1; d < SpaceDimension; ++d)
      {
        nonZeroJacobianIndices[d * nweights + i] += to_add;
      }
    }
  }
} // end GetJacobian()


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetSpatialJacobian(
  const InputPointType & inputPoint,
  SpatialJacobianType &  sj) const
{
  if (this->GetNumberOfParameters() == 0)
  {
    sj.SetIdentity();
    return;
  }

  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  int lidx = 0;
  PointToLabel(inputPoint, lidx);
  if (lidx == 0)
  {
    sj.SetIdentity();
    return;
  }
  SpatialJacobianType nsj;
  m_Trans[0]->GetSpatialJacobian(inputPoint, nsj);
  m_Trans[lidx]->GetSpatialJacobian(inputPoint, sj);
  sj += nsj;
}


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetSpatialHessian(
  const InputPointType & inputPoint,
  SpatialHessianType &   sh) const
{
  if (this->GetNumberOfParameters() == 0)
  {
    for (unsigned int i = 0; i < sh.Size(); ++i)
    {
      sh[i].Fill(0.0);
    }
    return;
  }

  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  int lidx = 0;
  PointToLabel(inputPoint, lidx);
  if (lidx == 0)
  {
    for (unsigned int i = 0; i < sh.Size(); ++i)
    {
      sh[i].Fill(0.0);
    }
    return;
  }

  SpatialHessianType nsh, lsh;
  m_Trans[0]->GetSpatialHessian(inputPoint, nsh);
  m_Trans[lidx]->GetSpatialHessian(inputPoint, lsh);
  for (unsigned i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned j = 0; j < SpaceDimension; ++j)
    {
      for (unsigned k = 0; k < SpaceDimension; ++k)
      {
        sh[i][j][k] = lsh[i][j][k] + nsh[i][j][k];
      }
    }
  }
}


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  itkExceptionMacro(<< "ERROR: GetJacobianOfSpatialJacobian() not yet implemented in the "
                       "MultiBSplineDeformableTransformWithNormal class.");
} // end GetJacobianOfSpatialJacobian()


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  if (this->GetNumberOfParameters() == 0)
  {
    jsj.resize(0);
    nonZeroJacobianIndices.resize(0);
    return;
  }

  // Initialize
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  jsj.resize(nnzji);

  // This implements a sparse version of the Jacobian.
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  int lidx = 0;
  PointToLabel(inputPoint, lidx);

  // Convert the physical point to a continuous index, which
  // is needed for the 'Evaluate()' functions below.
  const typename TransformType::ContinuousIndexType cindex =
    m_Trans[lidx]->TransformPointToContinuousGridIndex(inputPoint);

  if (lidx == 0 || !m_Trans[lidx]->InsideValidRegion(cindex))
  {
    sj.SetIdentity();
    for (unsigned int i = 0; i < jsj.size(); ++i)
    {
      jsj[i].Fill(0.0);
    }
    nonZeroJacobianIndices.resize(nnzji);
    for (unsigned int i = 0; i < nnzji; ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  SpatialJacobianType           nsj, lsj;
  JacobianOfSpatialJacobianType njsj, ljsj;

  // nzji should be the same so keep only one
  m_Trans[0]->GetJacobianOfSpatialJacobian(inputPoint, nsj, njsj, nonZeroJacobianIndices);
  m_Trans[lidx]->GetJacobianOfSpatialJacobian(inputPoint, lsj, ljsj, nonZeroJacobianIndices);

  using BaseContainer = typename ImageBaseType::PixelContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();

  const unsigned nweights = this->GetNumberOfWeights();
  for (unsigned i = 0; i < nweights; ++i)
  {
    VectorType tmp = bases[nonZeroJacobianIndices[i]][0];
    for (unsigned j = 0; j < SpaceDimension; ++j)
    {
      for (unsigned k = 0; k < SpaceDimension; ++k)
      {
        jsj[j][i][k] = tmp[j] * njsj[j][i + j * nweights][k];
      }
    }

    for (unsigned d = 1; d < SpaceDimension; ++d)
    {
      tmp = bases[nonZeroJacobianIndices[i]][d];
      for (unsigned j = 0; j < SpaceDimension; ++j)
      {
        for (unsigned k = 0; k < SpaceDimension; ++k)
        {
          jsj[j][i + d * nweights][k] = tmp[j] * ljsj[j][i + j * nweights][k];
        }
      }
    }
    sj = nsj + lsj;
  }

  // move non zero indices to match label positions
  if (lidx > 1)
  {
    unsigned to_add = (lidx - 1) * m_Trans[0]->GetNumberOfParametersPerDimension() * (SpaceDimension - 1);
    for (unsigned i = 0; i < nweights; ++i)
    {
      for (unsigned d = 1; d < SpaceDimension; ++d)
      {
        nonZeroJacobianIndices[d * nweights + i] += to_add;
      }
    }
  }
}


template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  if (this->GetNumberOfParameters() == 0)
  {
    jsh.resize(0);
    nonZeroJacobianIndices.resize(0);
    return;
  }

  // Initialize
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  jsh.resize(nnzji);

  // This implements a sparse version of the Jacobian.
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  int lidx = 0;
  PointToLabel(inputPoint, lidx);

  // Convert the physical point to a continuous index, which
  // is needed for the 'Evaluate()' functions below.
  const typename TransformType::ContinuousIndexType cindex =
    m_Trans[lidx]->TransformPointToContinuousGridIndex(inputPoint);

  if (lidx == 0 || !m_Trans[lidx]->InsideValidRegion(cindex))
  {
    // Return some dummy
    for (unsigned int i = 0; i < jsh.size(); ++i)
    {
      for (unsigned int j = 0; j < jsh[i].Size(); ++j)
      {
        jsh[i][j].Fill(0.0);
      }
    }
    for (unsigned int i = 0; i < sh.Size(); ++i)
    {
      sh[i].Fill(0.0);
    }
    nonZeroJacobianIndices.resize(nnzji);
    for (unsigned int i = 0; i < nnzji; ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  SpatialHessianType           nsh, lsh;
  JacobianOfSpatialHessianType njsh, ljsh;

  // nzji should be the same so keep only one
  m_Trans[0]->GetJacobianOfSpatialHessian(inputPoint, nsh, njsh, nonZeroJacobianIndices);
  m_Trans[lidx]->GetJacobianOfSpatialHessian(inputPoint, lsh, ljsh, nonZeroJacobianIndices);

  using BaseContainer = typename ImageBaseType::PixelContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();

  const unsigned nweights = this->GetNumberOfWeights();
  for (unsigned i = 0; i < nweights; ++i)
  {
    VectorType tmp = bases[nonZeroJacobianIndices[i]][0];
    for (unsigned j = 0; j < SpaceDimension; ++j)
    {
      for (unsigned k = 0; k < SpaceDimension; ++k)
      {
        for (unsigned l = 0; l < SpaceDimension; ++l)
        {
          jsh[j][i][k][l] = tmp[j] * njsh[j][i + j * nweights][k][l];
        }
      }
    }

    for (unsigned l = 1; l <= m_NbLabels; ++l)
    {
      VectorType tmp = bases[nonZeroJacobianIndices[i]][l];
      for (unsigned j = 0; j < SpaceDimension; ++j)
      {
        for (unsigned k = 0; k < SpaceDimension; ++k)
        {
          for (unsigned l = 0; l < SpaceDimension; ++l)
          {
            jsh[j][i + l * nweights][k][l] = tmp[j] * ljsh[j][i + j * nweights][k][l];
          }
        }
      }
    }
  }

  for (unsigned i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned j = 0; j < SpaceDimension; ++j)
    {
      for (unsigned k = 0; k < SpaceDimension; ++k)
      {
        sh[i][j][k] = lsh[i][j][k] + nsh[i][j][k];
      }
    }
  }

  // move non zero indices to match label positions
  if (lidx > 1)
  {
    unsigned to_add = (lidx - 1) * m_Trans[0]->GetNumberOfParametersPerDimension() * (SpaceDimension - 1);
    for (unsigned i = 0; i < nweights; ++i)
    {
      for (unsigned d = 1; d < SpaceDimension; ++d)
      {
        nonZeroJacobianIndices[d * nweights + i] += to_add;
      }
    }
  }
}


} // end namespace itk

#endif // end #ifndef itkMultiBSplineDeformableTransformWithNormal_hxx
