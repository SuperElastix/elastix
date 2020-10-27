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
#ifndef __itkANNStandardTreeSearch_h
#define __itkANNStandardTreeSearch_h

#include "itkBinaryANNTreeSearchBase.h"

namespace itk
{

/**
 * \class ANNStandardTreeSearch
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ANNStandardTreeSearch : public BinaryANNTreeSearchBase<TListSample>
{
public:
  /** Standard itk. */
  typedef ANNStandardTreeSearch                Self;
  typedef BinaryANNTreeSearchBase<TListSample> Superclass;
  typedef SmartPointer<Self>                   Pointer;
  typedef SmartPointer<const Self>             ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNStandardTreeSearch, BinaryANNTreeSearchBase);

  /** Typedefs from Superclass. */
  typedef typename Superclass::ListSampleType        ListSampleType;
  typedef typename Superclass::BinaryTreeType        BinaryTreeType;
  typedef typename Superclass::MeasurementVectorType MeasurementVectorType;
  typedef typename Superclass::IndexArrayType        IndexArrayType;
  typedef typename Superclass::DistanceArrayType     DistanceArrayType;

  typedef typename Superclass::ANNPointType         ANNPointType;         // double *
  typedef typename Superclass::ANNIndexType         ANNIndexType;         // int
  typedef typename Superclass::ANNIndexArrayType    ANNIndexArrayType;    // int *
  typedef typename Superclass::ANNDistanceType      ANNDistanceType;      // double
  typedef typename Superclass::ANNDistanceArrayType ANNDistanceArrayType; // double *

  typedef typename Superclass::BinaryANNTreeType BinaryANNTreeType;

  /** Set and get the error bound eps. */
  itkSetClampMacro(ErrorBound, double, 0.0, 1e14);
  itkGetConstMacro(ErrorBound, double);

  /** Search the nearest neighbours of a query point qp. */
  void
  Search(const MeasurementVectorType & qp, IndexArrayType & ind, DistanceArrayType & dists) override;

protected:
  ANNStandardTreeSearch();
  ~ANNStandardTreeSearch() override;

  /** Member variables. */
  double m_ErrorBound;

private:
  ANNStandardTreeSearch(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkANNStandardTreeSearch.hxx"
#endif

#endif // end #ifndef __itkANNStandardTreeSearch_h
