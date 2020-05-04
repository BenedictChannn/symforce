//  -----------------------------------------------------------------------------
// This file was autogenerated by symforce. Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include "./rot2.h"

// Explicit instantiation
template class geo::Rot2<double>;
template class geo::Rot2<float>;

// Print implementations
std::ostream& operator<<(std::ostream& os, const geo::Rot2d& a) {
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    os << "<Rot2d " << a.Data().transpose().format(fmt) << ">";
    return os;
}
std::ostream& operator<<(std::ostream& os, const geo::Rot2f& a) {
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    os << "<Rot2f " << a.Data().transpose().format(fmt) << ">";
    return os;
}


// Concept implementations for this class
#include "./ops/impl/rot2/storage_ops.cc"
#include "./ops/impl/rot2/group_ops.cc"
#include "./ops/impl/rot2/lie_group_ops.cc"