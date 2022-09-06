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
#ifndef itkANNFixedRadiusTreeSearch_h
#define itkANNFixedRadiusTreeSearch_h

#include "itkBinaryANNTreeSearchBase.h"

namespace itk
{

/**
 * \class ANNFixedRadiusTreeSearch
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT ANNFixedRadiusTreeSearch : public BinaryANNTreeSearchBase<TListSample>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ANNFixedRadiusTreeSearch);

  /** Standard itk. */
  using Self = ANNFixedRadiusTreeSearch;
  using Superclass = BinaryANNTreeSearchBase<TListSample>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNFixedRadiusTreeSearch, BinaryANNTreeSearchBase);

  /** Typedef's from Superclass. */
  using typename Superclass::ListSampleType;
  using typename Superclass::BinaryTreeType;
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::IndexArrayType;
  using typename Superclass::DistanceArrayType;

  using typename Superclass::ANNPointType;         // double *
  using typename Superclass::ANNIndexType;         // int
  using typename Superclass::ANNIndexArrayType;    // int *
  using typename Superclass::ANNDistanceType;      // double
  using typename Superclass::ANNDistanceArrayType; // double *

  using typename Superclass::BinaryANNTreeType;

  /** Set and get the error bound eps. */
  itkSetClampMacro(ErrorBound, double, 0.0, 1e14);
  itkGetConstMacro(ErrorBound, double);

  /** Set and get the squared radius search bound. */
  itkSetMacro(SquaredRadius, double);
  itkGetConstMacro(SquaredRadius, double);

  /** Search the nearest neighbours of a query point qp. */
  void
  Search(const MeasurementVectorType & qp, IndexArrayType & ind, DistanceArrayType & dists) override;

  /** Search the nearest neighbours of a query point qp. */
  virtual void
  Search(const MeasurementVectorType & qp, IndexArrayType & ind, DistanceArrayType & dists, double sqRad);

protected:
  ANNFixedRadiusTreeSearch();
  ~ANNFixedRadiusTreeSearch() override = default;

  /** Member variables. */
  double m_ErrorBound;
  double m_SquaredRadius;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkANNFixedRadiusTreeSearch.hxx"
#endif

#endif // end #ifndef itkANNFixedRadiusTreeSearch_h
